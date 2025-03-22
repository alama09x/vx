#![warn(clippy::all)]

use std::{
    borrow::Cow,
    collections::HashSet,
    error::Error,
    ffi::{c_void, CStr, CString},
    fs::File,
    io::{self, Read},
    mem,
    os::raw,
    path::Path,
    ptr, slice, str,
    time::Instant,
};

use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    prelude::VkResult,
    vk,
};
use data::{camera::CameraRaw, transform::Transform, DynIntoRaw};
use winit::{
    dpi::LogicalSize,
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowAttributes},
};

use data::camera::Camera;

#[allow(dead_code)]
pub struct VxState {
    pub window: Window,
    entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,
    device: ash::Device,
    queues: Queues,

    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_buffers: Vec<vk::CommandBuffer>,
    sync_objects: SyncObjects,

    current_frame: u8,

    start_time: Instant,
    current_time: Instant,
    previous_time: Instant,
    elapsed_time: f32,
    pub delta_time: f32,

    pub camera: Camera,
}

const MAX_FRAMES_IN_FLIGHT: u8 = 2;

impl VxState {
    const ENGINE_NAME: &str = "VX Engine";
    const ENGINE_VERSION: u32 = 0;
    const API_VERSION: u32 = vk::API_VERSION_1_3;

    const LAYER_NAMES: &[&CStr] = &[c"VK_LAYER_KHRONOS_validation"];
    const DEVICE_EXTENSION_NAMES: &[&CStr] = &[
        swapchain::NAME,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ];

    const VERTICES: [Vertex; 4] = [
        Vertex {
            pos: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: [0.0, -0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: [-0.5, -0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-0.5, 0.0, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    const INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

    pub fn new(
        app_name: &'static str,
        app_version: u32,
        window_title: &str,
        window_size: &LogicalSize<f32>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let window = Self::create_window(window_title, window_size, event_loop)?;
            let entry = ash::Entry::load()?;
            let instance = Self::create_instance(&entry, app_name, app_version, event_loop)?;

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_messenger = Self::create_debug_messenger(&debug_utils_loader)?;

            let surface_loader = surface::Instance::new(&entry, &instance);
            let surface = Self::create_surface(&entry, &instance, &window)?;

            let (physical_device, queue_family_indices) =
                Self::pick_physical_device(&instance, &surface_loader, surface)?;

            let (device, queues) =
                Self::create_logical_device(&instance, physical_device, &queue_family_indices)?;

            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) =
                Self::create_swapchain(
                    physical_device,
                    &surface_loader,
                    surface,
                    &swapchain_loader,
                    &window,
                    &queue_family_indices,
                )?;

            let swapchain_image_views =
                Self::create_image_views(&device, &swapchain_images, swapchain_image_format)?;

            let render_pass = Self::create_render_pass(&device, swapchain_image_format)?;

            let swapchain_framebuffers = Self::create_framebuffers(
                &device,
                render_pass,
                &swapchain_image_views,
                &swapchain_extent,
            )?;

            let descriptor_set_layout = Self::create_descriptor_set_layout(&device)?;
            let (pipeline_layout, pipeline) =
                Self::create_graphics_pipeline(&device, render_pass, descriptor_set_layout)?;

            let command_pool =
                Self::create_command_pool(&device, queue_family_indices.graphics_family)?;

            let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
                &instance,
                &device,
                physical_device,
                command_pool,
                queues.graphics,
            )?;

            let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
                &instance,
                &device,
                physical_device,
                command_pool,
                queues.graphics,
            )?;

            let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
                Self::create_uniform_buffers(
                    &instance,
                    &device,
                    physical_device,
                    MAX_FRAMES_IN_FLIGHT,
                )?;

            let descriptor_pool = Self::create_descriptor_pool(&device)?;
            let descriptor_sets = Self::create_descriptor_sets(
                &device,
                descriptor_pool,
                descriptor_set_layout,
                &uniform_buffers,
            )?;

            let command_buffers = Self::create_command_buffers(&device, command_pool)?;

            let sync_objects = SyncObjects::new(&device)?;

            let current_time = Instant::now();

            let camera = Camera::new(
                Transform::from_xyz(0.0, 0.0, 16.0),
                window_size.width,
                window_size.height,
            );

            Ok(Self {
                window,
                entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                surface,
                surface_loader,
                physical_device,
                queue_family_indices,
                device,
                queues,

                swapchain_loader,
                swapchain,
                swapchain_images,
                swapchain_image_format,
                swapchain_extent,
                swapchain_image_views,
                swapchain_framebuffers,

                render_pass,
                descriptor_set_layout,
                pipeline_layout,
                graphics_pipeline: pipeline,

                command_pool,

                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,

                uniform_buffers,
                uniform_buffers_memory,
                uniform_buffers_mapped,

                descriptor_pool,
                descriptor_sets,

                command_buffers,

                sync_objects,

                current_frame: 0,

                start_time: current_time,
                current_time,
                previous_time: current_time,
                elapsed_time: 0.0,
                delta_time: 0.0,

                camera,
            })
        }
    }

    fn update_time(&mut self) {
        self.current_time = Instant::now();
        self.elapsed_time = self
            .current_time
            .duration_since(self.start_time)
            .as_secs_f32();
        self.delta_time = self
            .current_time
            .duration_since(self.previous_time)
            .as_secs_f32();
        self.previous_time = self.current_time;
    }

    unsafe fn update_uniform_buffers(&mut self) -> VkResult<()> {
        ptr::copy_nonoverlapping(
            self.camera.to_bytes().as_ptr() as *const c_void,
            self.uniform_buffers_mapped[self.current_frame as usize],
            mem::size_of::<CameraRaw>(),
        );

        Ok(())
    }

    pub fn draw_frame(&mut self) -> VkResult<()> {
        unsafe {
            self.update_time();

            self.device.wait_for_fences(
                &[self.sync_objects.in_flight_fences[self.current_frame as usize]],
                true,
                u64::MAX,
            )?;

            let (image_index, _suboptimal) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.sync_objects.image_available_semaphores[self.current_frame as usize],
                vk::Fence::null(),
            ) {
                Ok(i) => i,
                Err(vk::Result::SUBOPTIMAL_KHR) => return Ok(()),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            self.update_uniform_buffers()?;

            self.device
                .reset_fences(&[self.sync_objects.in_flight_fences[self.current_frame as usize]])?;

            self.device.reset_command_buffer(
                self.command_buffers[self.current_frame as usize],
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.record_command_buffer(
                self.command_buffers[self.current_frame as usize],
                image_index,
            )?;

            let wait_semaphores =
                &[self.sync_objects.image_available_semaphores[self.current_frame as usize]];
            let signal_semaphores =
                &[self.sync_objects.render_finished_semaphores[self.current_frame as usize]];

            self.device.queue_submit(
                self.queues.graphics,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[self.command_buffers[self.current_frame as usize]])
                    .signal_semaphores(signal_semaphores)],
                self.sync_objects.in_flight_fences[self.current_frame as usize],
            )?;

            match self.swapchain_loader.queue_present(
                self.queues.present,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(signal_semaphores)
                    .swapchains(&[self.swapchain])
                    .image_indices(&[image_index]),
            ) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(e) => return Err(e),
            };
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

            Ok(())
        }
    }

    pub fn recreate_swapchain(&mut self) -> VkResult<()> {
        unsafe {
            self.device.device_wait_idle()?;
            let physical_size = self.window.inner_size();
            if physical_size.width == 0 || physical_size.height == 0 {
                return Ok(());
            }

            self.cleanup_swapchain();
            (
                self.swapchain,
                self.swapchain_images,
                self.swapchain_image_format,
                self.swapchain_extent,
            ) = Self::create_swapchain(
                self.physical_device,
                &self.surface_loader,
                self.surface,
                &self.swapchain_loader,
                &self.window,
                &self.queue_family_indices,
            )?;

            self.swapchain_image_views = Self::create_image_views(
                &self.device,
                &self.swapchain_images,
                self.swapchain_image_format,
            )?;
            self.swapchain_framebuffers = Self::create_framebuffers(
                &self.device,
                self.render_pass,
                &self.swapchain_image_views,
                &self.swapchain_extent,
            )?;

            Ok(())
        }
    }

    unsafe fn record_command_buffer(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
    ) -> VkResult<()> {
        self.device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

        self.device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.swapchain_framebuffers[image_index as usize])
                .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
                .clear_values(&[vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }]),
            vk::SubpassContents::INLINE,
        );

        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline,
        );

        self.device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(self.swapchain_extent.width as f32)
                .height(self.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)],
        );

        self.device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D::default().extent(self.swapchain_extent)],
        );

        self.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer,
            0,
            vk::IndexType::UINT16,
        );
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.descriptor_sets[self.current_frame as usize]],
            &[],
        );

        self.device
            .cmd_draw_indexed(command_buffer, Self::INDICES.len() as u32, 1, 0, 0, 0);

        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    fn create_window(
        title: &str,
        size: &LogicalSize<f32>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Window, Box<dyn Error>> {
        let window = event_loop.create_window(
            WindowAttributes::default()
                .with_title(title)
                .with_inner_size(*size),
        )?;
        window.request_redraw();
        Ok(window)
    }

    unsafe fn create_instance(
        entry: &ash::Entry,
        app_name: &str,
        app_version: u32,
        event_loop: &ActiveEventLoop,
    ) -> Result<ash::Instance, Box<dyn Error>> {
        let mut extension_names =
            ash_window::enumerate_required_extensions(event_loop.display_handle()?.as_raw())?
                .to_vec();
        extension_names.push(debug_utils::NAME.as_ptr());
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            // Enabling this extension is required when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(
                    &vk::ApplicationInfo::default()
                        .application_name(&CString::new(app_name).unwrap())
                        .application_version(app_version)
                        .engine_name(&CString::new(Self::ENGINE_NAME).unwrap())
                        .engine_version(Self::ENGINE_VERSION)
                        .api_version(Self::API_VERSION),
                )
                .enabled_layer_names(
                    &Self::LAYER_NAMES
                        .iter()
                        .map(|name| name.as_ptr())
                        .collect::<Vec<_>>(),
                )
                .enabled_extension_names(&extension_names)
                .flags(if cfg!(any(target_os = "macos", target_os = "ios")) {
                    vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
                } else {
                    vk::InstanceCreateFlags::default()
                }),
            None,
        )?;
        Ok(instance)
    }

    unsafe fn create_debug_messenger(
        debug_utils_loader: &debug_utils::Instance,
    ) -> VkResult<vk::DebugUtilsMessengerEXT> {
        debug_utils_loader.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback)),
            None,
        )
    }

    unsafe fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &winit::window::Window,
    ) -> Result<vk::SurfaceKHR, Box<dyn Error>> {
        let surface = ash_window::create_surface(
            entry,
            instance,
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            None,
        )?;
        Ok(surface)
    }

    unsafe fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices), Box<dyn Error>> {
        instance
            .enumerate_physical_devices()?
            .iter()
            .find_map(|&physical_device| {
                let indices =
                    Self::device_is_suitable(physical_device, instance, surface_loader, surface)
                        .ok()?;
                indices.map(|indices| (physical_device, indices))
            })
            .ok_or(Box::new(vk::Result::ERROR_UNKNOWN))
    }

    unsafe fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> VkResult<bool> {
        let available_extensions =
            instance.enumerate_device_extension_properties(physical_device)?;
        let mut required_extensions: HashSet<_> = Self::DEVICE_EXTENSION_NAMES
            .iter()
            .map(|e| e.to_bytes_with_nul().to_vec())
            .collect();

        for ext in available_extensions.iter() {
            if let Ok(ext_name) = ext.extension_name_as_c_str() {
                required_extensions.remove(ext_name.to_bytes_with_nul());
            }
        }

        Ok(required_extensions.is_empty())
    }

    /// Returns `Some(QueueFamilyIndices)` if the device is suitable
    unsafe fn device_is_suitable(
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Option<QueueFamilyIndices>> {
        let indices = QueueFamilyIndices::new(instance, physical_device, surface_loader, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, physical_device)?;
        let swapchain_adequate = {
            let swapchain_support =
                SwapchainSupportDetails::new(physical_device, surface_loader, surface)?;
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        };
        let supported_features = instance.get_physical_device_features(physical_device);

        if extensions_supported && swapchain_adequate && supported_features.sampler_anisotropy != 0
        {
            Ok(Some(indices))
        } else {
            Ok(None)
        }
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
    ) -> VkResult<(ash::Device, Queues)> {
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(
                    // Unique queue family indices
                    &queue_family_indices
                        .all()
                        .iter()
                        .collect::<HashSet<_>>()
                        .iter()
                        .map(|&&index| {
                            vk::DeviceQueueCreateInfo::default()
                                .queue_family_index(index)
                                .queue_priorities(&[1.0])
                        })
                        .collect::<Vec<_>>(),
                )
                .enabled_extension_names(
                    // Raw pointer extension names
                    &Self::DEVICE_EXTENSION_NAMES
                        .iter()
                        .map(|x| x.as_ptr())
                        .collect::<Vec<_>>(),
                )
                .enabled_features(&vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true)),
            None,
        )?;
        let queues = Queues {
            graphics: device.get_device_queue(queue_family_indices.graphics_family, 0),
            present: device.get_device_queue(queue_family_indices.present_family, 0),
        };
        Ok((device, queues))
    }

    fn choose_swapchain_surface_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> Option<&vk::SurfaceFormatKHR> {
        formats.iter().find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
    }

    fn choose_swapchain_present_mode(
        present_modes: &[vk::PresentModeKHR],
    ) -> Option<&vk::PresentModeKHR> {
        present_modes
            .iter()
            .find(|p| **p == vk::PresentModeKHR::MAILBOX || **p == vk::PresentModeKHR::FIFO)
    }

    fn choose_swapchain_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &Window,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    unsafe fn create_swapchain(
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
        swapchain_loader: &swapchain::Device,
        window: &Window,
        queue_familiy_indices: &QueueFamilyIndices,
    ) -> VkResult<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
        let SwapchainSupportDetails {
            capabilities,
            formats,
            present_modes,
        } = SwapchainSupportDetails::new(physical_device, surface_loader, surface)?;

        let surface_format =
            Self::choose_swapchain_surface_format(&formats).ok_or(vk::Result::ERROR_UNKNOWN)?;

        let present_mode =
            Self::choose_swapchain_present_mode(&present_modes).ok_or(vk::Result::ERROR_UNKNOWN)?;

        let extent = Self::choose_swapchain_extent(&capabilities, window);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.min_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let unique_indices: Vec<_> = queue_familiy_indices
            .all()
            .iter()
            .collect::<HashSet<_>>()
            .iter()
            .map(|x| **x)
            .collect();

        let swapchain = swapchain_loader.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(if unique_indices.len() == 1 {
                    vk::SharingMode::EXCLUSIVE
                } else {
                    vk::SharingMode::CONCURRENT
                })
                .queue_family_indices(&unique_indices)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(*present_mode)
                .clipped(true),
            None,
        )?;

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain)?;

        Ok((swapchain, swapchain_images, surface_format.format, extent))
    }

    unsafe fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> VkResult<vk::ImageView> {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                ),
            None,
        )
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> VkResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| Self::create_image_view(device, image, format))
            .collect()
    }

    unsafe fn cleanup_swapchain(&self) {
        for (&image_view, &framebuffer) in self
            .swapchain_image_views
            .iter()
            .zip(self.swapchain_framebuffers.iter())
        {
            self.device.destroy_framebuffer(framebuffer, None);
            self.device.destroy_image_view(image_view, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }

    unsafe fn create_descriptor_set_layout(
        device: &ash::Device,
    ) -> VkResult<vk::DescriptorSetLayout> {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX),
            ]),
            None,
        )
    }

    unsafe fn create_render_pass(
        device: &ash::Device,
        swapchain_image_format: vk::Format,
    ) -> VkResult<vk::RenderPass> {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&[vk::AttachmentDescription::default()
                    .format(swapchain_image_format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)])
                .subpasses(&[vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&[vk::AttachmentReference::default()
                        .attachment(0)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])])
                .dependencies(&[vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::from_raw(0))
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)]),
            None,
        )
    }

    unsafe fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        swapchain_image_views: &[vk::ImageView],
        swapchain_extent: &vk::Extent2D,
    ) -> VkResult<Vec<vk::Framebuffer>> {
        swapchain_image_views
            .iter()
            .map(|image_view| {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(slice::from_ref(image_view))
                        .width(swapchain_extent.width)
                        .height(swapchain_extent.height)
                        .layers(1),
                    None,
                )
            })
            .collect()
    }

    fn read_shader_code(path: &Path) -> io::Result<Vec<u32>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // SPIR-V uses 32-bit words
        if buffer.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SPIR-V binary size must be a multiple of 4 bytes",
            ));
        }

        let code: Vec<u32> = buffer
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        if code.is_empty() || code[0] != 0x07230203 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid SPIR-V binary: missing or incorrect magic number",
            ));
        }
        Ok(code)
    }

    unsafe fn create_shader_module(
        device: &ash::Device,
        code: &[u32],
    ) -> VkResult<vk::ShaderModule> {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
    }

    unsafe fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
        let vert_shader = Self::read_shader_code(Path::new("./bin/shader.vert.spv"))?;
        let frag_shader = Self::read_shader_code(Path::new("./bin/shader.frag.spv"))?;

        let vert_shader_module = Self::create_shader_module(device, &vert_shader)?;
        let frag_shader_module = Self::create_shader_module(device, &frag_shader)?;

        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
            None,
        )?;

        let pipelines = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(
                    &vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vert_shader_module)
                                .name(c"main"),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(frag_shader_module)
                                .name(c"main"),
                        ])
                        .vertex_input_state(
                            &vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(slice::from_ref(
                                    &Vertex::BINDING_DESCRIPTION,
                                ))
                                .vertex_attribute_descriptions(&Vertex::ATTRIBUTE_DESCRIPTIONS),
                        )
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .depth_clamp_enable(false)
                                .rasterizer_discard_enable(false)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .line_width(1.0)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .depth_bias_enable(false),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .sample_shading_enable(false)
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .logic_op_enable(false)
                                .logic_op(vk::LogicOp::COPY)
                                .attachments(slice::from_ref(
                                    &vk::PipelineColorBlendAttachmentState::default()
                                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                                        .blend_enable(false),
                                ))
                                .blend_constants([0.0, 0.0, 0.0, 0.0]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0),
                ),
                None,
            )
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;

        device.destroy_shader_module(frag_shader_module, None);
        device.destroy_shader_module(vert_shader_module, None);
        Ok((pipeline_layout, pipelines[0]))
    }

    unsafe fn create_command_pool(device: &ash::Device, family: u32) -> VkResult<vk::CommandPool> {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family),
            None,
        )
    }

    unsafe fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> VkResult<(u32, vk::MemoryType)> {
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        memory_properties
            .memory_types
            .iter()
            .enumerate()
            .find_map(|(i, memory_type)| {
                if (type_filter & (1 << i)) != 0
                    && (memory_type.property_flags & properties) == properties
                {
                    Some((i as u32, *memory_type))
                } else {
                    None
                }
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)
    }

    unsafe fn create_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?; // TODO: check `EXCLUSIVE`

        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    Self::find_memory_type(
                        instance,
                        physical_device,
                        memory_requirements.memory_type_bits,
                        properties,
                    )?
                    .0,
                ),
            None,
        )?;

        device.bind_buffer_memory(buffer, memory, 0)?;
        Ok((buffer, memory))
    }

    unsafe fn begin_single_time_commands(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> VkResult<vk::CommandBuffer> {
        let command_buffer = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1),
        )?[0];

        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        Ok(command_buffer)
    }

    unsafe fn end_single_time_commands(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        command_buffer: vk::CommandBuffer,
        graphics_queue: vk::Queue,
    ) -> VkResult<()> {
        device.end_command_buffer(command_buffer)?;

        device.queue_submit(
            graphics_queue,
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            vk::Fence::null(),
        )?;

        device.queue_wait_idle(graphics_queue)?;

        device.free_command_buffers(command_pool, &[command_buffer]);

        Ok(())
    }

    unsafe fn copy_buffer(
        device: &ash::Device,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) -> VkResult<()> {
        let command_buffer = Self::begin_single_time_commands(device, command_pool)?;

        device.cmd_copy_buffer(
            command_buffer,
            src,
            dst,
            &[vk::BufferCopy::default().size(size)],
        );

        Self::end_single_time_commands(device, command_pool, command_buffer, graphics_queue)?;

        Ok(())
    }

    unsafe fn create_buffer_from_data_with_staging<T>(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        data: &[T],
        buffer_usage: vk::BufferUsageFlags,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_size = mem::size_of_val(data) as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let buffer_data = device.map_memory(
            staging_buffer_memory,
            0,
            buffer_size,
            vk::MemoryMapFlags::from_raw(0),
        )?;
        ptr::copy_nonoverlapping(
            data.as_ptr() as *const c_void,
            buffer_data,
            buffer_size as usize,
        );
        device.unmap_memory(staging_buffer_memory);

        let (buffer, buffer_memory) = Self::create_buffer(
            instance,
            device,
            physical_device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | buffer_usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        Self::copy_buffer(
            device,
            command_pool,
            graphics_queue,
            staging_buffer,
            buffer,
            buffer_size,
        )?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok((buffer, buffer_memory))
    }

    unsafe fn create_vertex_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        Self::create_buffer_from_data_with_staging(
            instance,
            device,
            physical_device,
            command_pool,
            graphics_queue,
            &Self::VERTICES,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
    }

    unsafe fn create_index_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        Self::create_buffer_from_data_with_staging(
            instance,
            device,
            physical_device,
            command_pool,
            graphics_queue,
            &Self::INDICES,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )
    }

    unsafe fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        frames: u8,
    ) -> VkResult<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>)> {
        let buffer_size = mem::size_of::<CameraRaw>() as u64;

        let mut buffers = Vec::with_capacity(frames as usize);
        let mut memory = Vec::with_capacity(frames as usize);
        let mut mapped = Vec::with_capacity(frames as usize);

        for _ in 0..frames as usize {
            let (buffer, mem) = Self::create_buffer(
                instance,
                device,
                physical_device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | { vk::MemoryPropertyFlags::HOST_COHERENT },
            )?;
            let map = device.map_memory(mem, 0, buffer_size, vk::MemoryMapFlags::empty())?;

            buffers.push(buffer);
            memory.push(mem);
            mapped.push(map);
        }

        Ok((buffers, memory, mapped))
    }

    unsafe fn create_descriptor_pool(device: &ash::Device) -> VkResult<vk::DescriptorPool> {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .pool_sizes(&[vk::DescriptorPoolSize::default()
                    .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)])
                .max_sets(MAX_FRAMES_IN_FLIGHT as u32),
            None,
        )
    }

    unsafe fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        let descriptor_sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize]),
        )?;

        for (frame, &descriptor_set) in descriptor_sets.iter().enumerate() {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(uniform_buffers[frame])
                        .offset(0)
                        .range(mem::size_of::<CameraRaw>() as u64)])],
                &[],
            );
        }

        Ok(descriptor_sets)
    }

    unsafe fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> VkResult<Vec<vk::CommandBuffer>> {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
        )
    }
}

struct SyncObjects {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
}

impl SyncObjects {
    pub unsafe fn new(device: &ash::Device) -> VkResult<Self> {
        let sync_objects: Vec<_> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                let image_sem = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None);
                let render_sem = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None);
                let in_flight_fence = device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                );
                (image_sem, render_sem, in_flight_fence)
            })
            .collect();

        Ok(Self {
            image_available_semaphores: sync_objects
                .iter()
                .map(|(s, _, _)| *s)
                .collect::<VkResult<Vec<_>>>()?,
            render_finished_semaphores: sync_objects
                .iter()
                .map(|(_, s, _)| *s)
                .collect::<VkResult<Vec<_>>>()?,
            in_flight_fences: sync_objects
                .iter()
                .map(|(_, _, f)| *f)
                .collect::<VkResult<Vec<_>>>()?,
        })
    }
}

struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const BINDING_DESCRIPTION: vk::VertexInputBindingDescription =
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };

    pub const ATTRIBUTE_DESCRIPTIONS: [vk::VertexInputAttributeDescription; 2] = [
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: mem::offset_of!(Self, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: mem::offset_of!(Self, color) as u32,
        },
    ];
}

struct QueueFamilyIndices {
    graphics_family: u32,
    present_family: u32,
}

impl QueueFamilyIndices {
    pub const fn all(&self) -> [u32; 2] {
        [self.graphics_family, self.present_family]
    }
}

struct Queues {
    graphics: vk::Queue,
    present: vk::Queue,
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub unsafe fn new(
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Self> {
        let capabilities =
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;

        let formats =
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?;

        let present_modes =
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?;

        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }
}

impl QueueFamilyIndices {
    pub unsafe fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Self> {
        let queue_families = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics_family = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, properties)| {
                if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)?;

        let present_family = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, _)| {
                if surface_loader
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .ok()?
                {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)?;

        Ok(Self {
            graphics_family,
            present_family,
        })
    }
}

impl Drop for VxState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
                self.device
                    .destroy_semaphore(self.sync_objects.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.sync_objects.render_finished_semaphores[i], None);
                self.device
                    .destroy_fence(self.sync_objects.in_flight_fences[i], None);
            }

            self.device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .unwrap();
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            for frame in 0..MAX_FRAMES_IN_FLIGHT as usize {
                self.device.unmap_memory(self.uniform_buffers_memory[frame]);
                self.device
                    .destroy_buffer(self.uniform_buffers[frame], None);
                self.device
                    .free_memory(self.uniform_buffers_memory[frame], None);
            }

            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.index_buffer_memory, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            self.cleanup_swapchain();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!("{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n");
    vk::FALSE
}
