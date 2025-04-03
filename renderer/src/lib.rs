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
};

use ash::{
    ext::{buffer_device_address, debug_utils},
    khr::{
        acceleration_structure, deferred_host_operations, get_physical_device_properties2,
        ray_tracing_pipeline, surface, swapchain,
    },
    prelude::VkResult,
    vk,
};
use bevy_ecs::system::Resource;
use data::{camera::CameraGpu, IntoBytes};

use glam::Vec2;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

const UNIFORM_BUFFER_SIZE: usize = mem::size_of::<CameraGpu>();

const VERTICES: [Vertex; 24] = [
    // Front
    Vertex {
        pos: [0.5, 0.5, 0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    // Back
    Vertex {
        pos: [-0.5, 0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [-0.5, -0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, -0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 0.0, 1.0],
    },
    // Bottom
    Vertex {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, 0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, -0.5],
        color: [1.0, 0.0, 1.0],
    },
    // Top
    Vertex {
        pos: [0.5, -0.5, 0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, -0.5, -0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, -0.5, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    // Right
    Vertex {
        pos: [0.5, 0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [0.5, 0.5, 0.5],
        color: [1.0, 0.0, 1.0],
    },
    // Left
    Vertex {
        pos: [-0.5, 0.5, 0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [-0.5, -0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, -0.5, -0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5, -0.5],
        color: [1.0, 0.0, 1.0],
    },
];

const INDICES: [u16; 6 * 6] = [
    0, 1, 2, 0, 2, 3, // Front
    4, 5, 6, 4, 6, 7, // Back
    8, 9, 10, 8, 10, 11, // Bottom
    12, 13, 14, 12, 14, 15, // Top
    16, 17, 18, 16, 18, 19, // Right
    20, 21, 22, 20, 22, 23, // Left
];

#[derive(Resource)]
pub struct InitState {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: QueueFamilyIndices,
    device: ash::Device,
    queues: Queues,
    command_pool: vk::CommandPool,
}

impl InitState {
    const ENGINE_NAME: &str = "VX Engine";
    const ENGINE_VERSION: u32 = 0;
    const API_VERSION: u32 = vk::API_VERSION_1_3;

    const LAYER_NAMES: &[&CStr] = &[c"VK_LAYER_KHRONOS_validation"];
    const DEVICE_EXTENSION_NAMES: &[&CStr] = &[
        swapchain::NAME,
        ray_tracing_pipeline::NAME,
        acceleration_structure::NAME,
        deferred_host_operations::NAME,
        get_physical_device_properties2::NAME,
        buffer_device_address::NAME,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ];

    pub fn new(
        app_name: &'static str,
        app_version: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;
            let instance = Self::create_instance(&entry, app_name, app_version, display_handle)?;

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_messenger = Self::create_debug_messenger(&debug_utils_loader)?;

            let surface_loader = surface::Instance::new(&entry, &instance);
            let surface = Self::create_surface(&entry, &instance, display_handle, window_handle)?;

            let (physical_device, queue_family_indices) =
                Self::pick_physical_device(&instance, &surface_loader, surface)?;

            let (device, queues) =
                Self::create_logical_device(&instance, physical_device, &queue_family_indices)?;

            let command_pool =
                Self::create_command_pool(&device, queue_family_indices.graphics_family)?;

            Ok(Self {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                surface_loader,
                surface,
                physical_device,
                queue_family_indices,
                device,
                queues,
                command_pool,
            })
        }
    }

    pub fn wait_idle(&self) -> VkResult<()> {
        unsafe { self.device.device_wait_idle()? }
        Ok(())
    }

    unsafe fn create_instance(
        entry: &ash::Entry,
        app_name: &str,
        app_version: u32,
        display_handle: RawDisplayHandle,
    ) -> Result<ash::Instance, Box<dyn Error>> {
        let mut extension_names =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();
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
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> VkResult<vk::SurfaceKHR> {
        ash_window::create_surface(entry, instance, display_handle, window_handle, None)
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
            transfer: device.get_device_queue(queue_family_indices.present_family, 0),
            present: device.get_device_queue(queue_family_indices.present_family, 0),
        };
        Ok((device, queues))
    }

    unsafe fn create_command_pool(device: &ash::Device, family: u32) -> VkResult<vk::CommandPool> {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family),
            None,
        )
    }
}

impl Drop for InitState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Resource)]
pub struct SwapchainRenderState {
    loader: swapchain::Device,
    image_format: vk::Format,
    extent: vk::Extent2D,

    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    output_image: vk::Image,
    output_image_memory: vk::DeviceMemory,
    output_image_view: vk::ImageView,
}

impl SwapchainRenderState {
    pub fn new(init_state: &InitState, window_size: Vec2) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let loader = swapchain::Device::new(&init_state.instance, &init_state.device);

            let (swapchain, image_format, extent, images) = Self::create_swapchain(
                init_state.physical_device,
                &init_state.surface_loader,
                init_state.surface,
                &init_state.queue_family_indices,
                &loader,
                window_size,
            )?;

            let image_views = Self::create_image_views(&init_state.device, image_format, &images)?;

            let render_pass = Self::create_render_pass(&init_state.device, image_format)?;
            let framebuffers =
                Self::create_framebuffers(&init_state.device, render_pass, &extent, &image_views)?;

            let (output_image, output_image_memory) = Self::create_output_image(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                extent,
            )?;

            let output_image_view =
                Self::create_image_view(&init_state.device, output_image, image_format)?;

            Ok(Self {
                loader,
                image_format,
                extent,

                swapchain,
                images,
                image_views,

                render_pass,
                framebuffers,

                output_image,
                output_image_memory,
                output_image_view,
            })
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        init_state: &InitState,
        window_size: Vec2,
    ) -> VkResult<()> {
        unsafe {
            init_state.device.device_wait_idle()?;
            if window_size.x == 0.0 || window_size.y == 0.0 {
                return Ok(());
            }

            self.cleanup_swapchain(init_state);
            (self.swapchain, self.image_format, self.extent, self.images) = Self::create_swapchain(
                init_state.physical_device,
                &init_state.surface_loader,
                init_state.surface,
                &init_state.queue_family_indices,
                &self.loader,
                window_size,
            )?;

            self.image_views =
                Self::create_image_views(&init_state.device, self.image_format, &self.images)?;
            self.framebuffers = Self::create_framebuffers(
                &init_state.device,
                self.render_pass,
                &self.extent,
                &self.image_views,
            )?;

            (self.output_image, self.output_image_memory) = Self::create_output_image(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                self.extent,
            )?;
            self.output_image_view =
                Self::create_image_view(&init_state.device, self.output_image, self.image_format)?;

            Ok(())
        }
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<&vk::SurfaceFormatKHR> {
        formats.iter().find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
    }

    fn choose_present_mode(present_modes: &[vk::PresentModeKHR]) -> Option<&vk::PresentModeKHR> {
        present_modes
            .iter()
            .find(|p| **p == vk::PresentModeKHR::MAILBOX || **p == vk::PresentModeKHR::FIFO)
    }

    fn choose_extent(capabilities: &vk::SurfaceCapabilitiesKHR, window_size: Vec2) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: (window_size.x.round() as u32).clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: (window_size.y.round() as u32).clamp(
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
        queue_familiy_indices: &QueueFamilyIndices,
        swapchain_loader: &swapchain::Device,
        window_size: Vec2,
    ) -> VkResult<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
        let SwapchainSupportDetails {
            capabilities,
            formats,
            present_modes,
        } = SwapchainSupportDetails::new(physical_device, surface_loader, surface)?;

        let surface_format =
            Self::choose_surface_format(&formats).ok_or(vk::Result::ERROR_UNKNOWN)?;

        let present_mode =
            Self::choose_present_mode(&present_modes).ok_or(vk::Result::ERROR_UNKNOWN)?;

        let extent = Self::choose_extent(&capabilities, window_size);

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

        Ok((swapchain, surface_format.format, extent, swapchain_images))
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
        format: vk::Format,
        images: &[vk::Image],
    ) -> VkResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| Self::create_image_view(device, image, format))
            .collect()
    }

    unsafe fn cleanup_swapchain(&self, init_state: &InitState) {
        for (&image_view, &framebuffer) in self.image_views.iter().zip(self.framebuffers.iter()) {
            init_state.device.destroy_framebuffer(framebuffer, None);
            init_state.device.destroy_image_view(image_view, None);
        }
        init_state
            .device
            .destroy_image_view(self.output_image_view, None);
        init_state.device.destroy_image(self.output_image, None);
        init_state
            .device
            .free_memory(self.output_image_memory, None);

        self.loader.destroy_swapchain(self.swapchain, None);
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
        swapchain_extent: &vk::Extent2D,
        swapchain_image_views: &[vk::ImageView],
    ) -> VkResult<Vec<vk::Framebuffer>> {
        swapchain_image_views
            .iter()
            .map(|&image_view| {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&[image_view])
                        .width(swapchain_extent.width)
                        .height(swapchain_extent.height)
                        .layers(1),
                    None,
                )
            })
            .collect()
    }

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            init_state
                .device
                .destroy_render_pass(self.render_pass, None);
            self.cleanup_swapchain(init_state);
        }
    }

    fn create_output_image(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
    ) -> VkResult<(vk::Image, vk::DeviceMemory)> {
        unsafe {
            let image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC),
                None,
            )?;

            let memory_requirements = device.get_image_memory_requirements(image);
            let (memory_type_index, _) = BuffersState::find_memory_type(
                instance,
                physical_device,
                memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index),
                None,
            )?;

            device.bind_image_memory(image, memory, 0)?;
            Ok((image, memory))
        }
    }
}

#[derive(Resource)]
pub struct PipelineState {
    ray_tracing_loader: ray_tracing_pipeline::Device,
    buffer_device_address_loader: buffer_device_address::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_binding_table: ShaderBindingTable,
}

impl PipelineState {
    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let ray_tracing_loader =
                ray_tracing_pipeline::Device::new(&init_state.instance, &init_state.device);
            let buffer_device_address_loader =
                buffer_device_address::Device::new(&init_state.instance, &init_state.device);

            let descriptor_set_layout = Self::create_descriptor_set_layout(&init_state.device)?;

            let (pipeline_layout, pipeline) = Self::create_pipeline(
                &init_state.device,
                &ray_tracing_loader,
                descriptor_set_layout,
            )?;

            let shader_binding_table = Self::create_shader_binding_table(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                &buffer_device_address_loader,
                &ray_tracing_loader,
                pipeline,
            )?;

            Ok(Self {
                ray_tracing_loader,
                buffer_device_address_loader,
                descriptor_set_layout,
                pipeline_layout,
                pipeline,
                shader_binding_table,
            })
        }
    }

    unsafe fn create_descriptor_set_layout(
        device: &ash::Device,
    ) -> VkResult<vk::DescriptorSetLayout> {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            ]),
            None,
        )
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

    unsafe fn create_pipeline(
        device: &ash::Device,
        ray_tracing_loader: &ray_tracing_pipeline::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
        let raygen_shader = Self::read_shader_code(Path::new("./bin/raygen.rgen.spv"))?;
        let miss_shader = Self::read_shader_code(Path::new("./bin/miss.rmiss.spv"))?;
        let closest_hit_shader = Self::read_shader_code(Path::new("./bin/closesthit.rchit.spv"))?;

        let raygen_module = Self::create_shader_module(device, &raygen_shader)?;
        let miss_module = Self::create_shader_module(device, &miss_shader)?;
        let closest_hit_module = Self::create_shader_module(device, &closest_hit_shader)?;

        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
            None,
        )?;

        let pipelines = ray_tracing_loader
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                            .module(raygen_module)
                            .name(c"main"),
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::MISS_KHR)
                            .module(miss_module)
                            .name(c"main"),
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .module(closest_hit_module)
                            .name(c"main"),
                    ])
                    .groups(&[
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(0)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(1)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .general_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(2)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                    ])
                    .max_pipeline_ray_recursion_depth(1)
                    .layout(pipeline_layout)],
                None,
            )
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;

        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(closest_hit_module, None);
        Ok((pipeline_layout, pipelines[0]))
    }

    unsafe fn create_shader_binding_table(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        bda_loader: &buffer_device_address::Device,
        rt_loader: &ray_tracing_pipeline::Device,
        pipeline: vk::Pipeline,
    ) -> Result<ShaderBindingTable, Box<dyn Error>> {
        let mut rt_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        instance.get_physical_device_properties2(
            physical_device,
            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut rt_properties),
        );

        let handle_size = rt_properties.shader_group_handle_size as vk::DeviceSize;
        let group_count = 3;
        let total_size = handle_size * group_count;

        let group_alignment = rt_properties.shader_group_handle_alignment as vk::DeviceSize;
        let aligned_size = (total_size + group_alignment - 1) & !(group_alignment - 1);

        let (buffer, memory) = BuffersState::create_buffer(
            instance,
            device,
            physical_device,
            aligned_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let handles = rt_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count as u32,
            total_size as usize,
        )?;

        let mapped =
            device.map_memory(memory, 0, aligned_size, vk::MemoryMapFlags::empty())? as *mut u8;
        ptr::copy_nonoverlapping(handles.as_ptr(), mapped, handles.len());
        device.unmap_memory(memory);

        let buffer_address = bda_loader
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

        let region_size = handle_size;

        Ok(ShaderBindingTable {
            buffer,
            memory,
            raygen_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(buffer_address)
                .stride(region_size)
                .size(region_size),
            miss_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(buffer_address + region_size)
                .stride(region_size)
                .size(region_size),
            hit_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(buffer_address + region_size * 2)
                .stride(region_size)
                .size(region_size),
        })
    }

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            init_state.device.destroy_pipeline(self.pipeline, None);
            init_state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            init_state
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            init_state
                .device
                .destroy_buffer(self.shader_binding_table.buffer, None);
            init_state
                .device
                .free_memory(self.shader_binding_table.memory, None);
        }
    }
}

struct ShaderBindingTable {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

#[derive(Resource)]
pub struct BuffersState {
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<&'static mut [u8]>,
}

impl BuffersState {
    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                init_state.command_pool,
                init_state.queues.transfer,
            )?;

            let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                init_state.command_pool,
                init_state.queues.transfer,
            )?;

            let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
                Self::create_uniform_buffers(
                    &init_state.instance,
                    &init_state.device,
                    init_state.physical_device,
                    MAX_FRAMES_IN_FLIGHT,
                )?;

            Ok(Self {
                vertex_buffer,
                vertex_buffer_memory,
                index_buffer,
                index_buffer_memory,

                uniform_buffers,
                uniform_buffers_memory,
                uniform_buffers_mapped,
            })
        }
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
        transfer_queue: vk::Queue,
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
            transfer_queue,
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
        transfer_queue: vk::Queue,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        Self::create_buffer_from_data_with_staging(
            instance,
            device,
            physical_device,
            command_pool,
            transfer_queue,
            &VERTICES,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )
    }

    unsafe fn create_index_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
    ) -> VkResult<(vk::Buffer, vk::DeviceMemory)> {
        Self::create_buffer_from_data_with_staging(
            instance,
            device,
            physical_device,
            command_pool,
            transfer_queue,
            &INDICES,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )
    }

    #[allow(clippy::type_complexity)]
    unsafe fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        frames: u8,
    ) -> VkResult<(
        Vec<vk::Buffer>,
        Vec<vk::DeviceMemory>,
        Vec<&'static mut [u8]>,
    )> {
        let buffer_size = mem::size_of::<CameraGpu>();

        let mut buffers = Vec::with_capacity(frames as usize);
        let mut memory = Vec::with_capacity(frames as usize);
        let mut mapped = Vec::with_capacity(frames as usize);

        for _ in 0..frames as usize {
            let (buffer, mem) = Self::create_buffer(
                instance,
                device,
                physical_device,
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | { vk::MemoryPropertyFlags::HOST_COHERENT },
            )?;
            let map_ptr =
                device.map_memory(mem, 0, buffer_size as u64, vk::MemoryMapFlags::empty())?
                    as *mut u8;
            let map = slice::from_raw_parts_mut(map_ptr, buffer_size);

            buffers.push(buffer);
            memory.push(mem);
            mapped.push(map);
        }

        Ok((buffers, memory, mapped))
    }

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            for frame in 0..MAX_FRAMES_IN_FLIGHT as usize {
                init_state
                    .device
                    .unmap_memory(self.uniform_buffers_memory[frame]);
                init_state
                    .device
                    .destroy_buffer(self.uniform_buffers[frame], None);
                init_state
                    .device
                    .free_memory(self.uniform_buffers_memory[frame], None);
            }

            init_state.device.destroy_buffer(self.index_buffer, None);
            init_state
                .device
                .free_memory(self.index_buffer_memory, None);

            init_state.device.destroy_buffer(self.vertex_buffer, None);
            init_state
                .device
                .free_memory(self.vertex_buffer_memory, None);
        }
    }
}

#[derive(Resource)]
pub struct AccelerationStructureState {
    acceleration_structure_loader: acceleration_structure::Device,
    blas: vk::AccelerationStructureKHR,
    blas_buffer: vk::Buffer,
    blas_memory: vk::DeviceMemory,
    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: vk::Buffer,
    tlas_memory: vk::DeviceMemory,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl AccelerationStructureState {
    pub fn new(
        init_state: &InitState,
        swapchain_render_state: &SwapchainRenderState,
        pipeline_state: &PipelineState,
        buffers_state: &BuffersState,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let acceleration_structure_loader =
                acceleration_structure::Device::new(&init_state.instance, &init_state.device);

            let (blas, blas_buffer, blas_memory) = Self::create_blas(
                &acceleration_structure_loader,
                init_state,
                pipeline_state,
                buffers_state,
            )?;
            let (tlas, tlas_buffer, tlas_memory) = Self::create_tlas(
                &acceleration_structure_loader,
                init_state,
                pipeline_state,
                blas,
            )?;

            let descriptor_pool = Self::create_descriptor_pool(&init_state.device)?;
            let descriptor_sets = Self::create_descriptor_sets(
                &init_state.device,
                descriptor_pool,
                pipeline_state.descriptor_set_layout,
                &buffers_state.uniform_buffers,
                tlas,
                &swapchain_render_state.output_image_view,
            )?;

            Ok(Self {
                acceleration_structure_loader,
                blas,
                blas_buffer,
                blas_memory,
                tlas,
                tlas_buffer,
                tlas_memory,
                descriptor_pool,
                descriptor_sets,
            })
        }
    }

    unsafe fn create_blas(
        acceleration_structure_loader: &acceleration_structure::Device,
        init_state: &InitState,
        pipeline_state: &PipelineState,
        buffers_state: &BuffersState,
    ) -> Result<(vk::AccelerationStructureKHR, vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        let geometries = &[vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: pipeline_state
                            .buffer_device_address_loader
                            .get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(buffers_state.index_buffer),
                            ),
                    }),
            })];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::HOST,
            &build_info,
            &[INDICES.len() as u32 / 3],
            &mut size_info,
        );

        let (buffer, memory) = BuffersState::create_buffer(
            &init_state.instance,
            &init_state.device,
            init_state.physical_device,
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let acceleration_structure = acceleration_structure_loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buffer)
                .size(size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?;

        let (scratch_buffer, scratch_memory) = BuffersState::create_buffer(
            &init_state.instance,
            &init_state.device,
            init_state.physical_device,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // TODO: create_buffer without memory map

        let scratch_address = pipeline_state
            .buffer_device_address_loader
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

        let command_buffer = init_state.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(init_state.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];

        init_state.device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        build_info = build_info
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

        acceleration_structure_loader.cmd_build_acceleration_structures(
            command_buffer,
            &[build_info],
            &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(INDICES.len() as u32 / 3)]],
        );

        init_state.device.end_command_buffer(command_buffer)?;

        init_state.device.queue_submit(
            init_state.queues.graphics,
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            vk::Fence::null(),
        )?;

        init_state
            .device
            .free_command_buffers(init_state.command_pool, &[command_buffer]);
        init_state.device.destroy_buffer(scratch_buffer, None);
        init_state.device.free_memory(scratch_memory, None);

        Ok((acceleration_structure, buffer, memory))
    }

    unsafe fn create_tlas(
        acceleration_structure_loader: &acceleration_structure::Device,
        init_state: &InitState,
        pipeline_state: &PipelineState,
        blas: vk::AccelerationStructureKHR,
    ) -> Result<(vk::AccelerationStructureKHR, vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        let instance = vk::AccelerationStructureInstanceKHR {
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: acceleration_structure_loader
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(blas),
                    ),
            },
            transform: vk::TransformMatrixKHR {
                #[rustfmt::skip]
                matrix: [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                ],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
        };

        let (instances_buffer, instances_memory) =
            BuffersState::create_buffer_from_data_with_staging(
                &init_state.instance,
                &init_state.device,
                init_state.physical_device,
                init_state.command_pool,
                init_state.queues.transfer,
                &[instance],
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            )?;

        let geometries = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: pipeline_state
                            .buffer_device_address_loader
                            .get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default().buffer(instances_buffer),
                            ),
                    },
                ),
            })];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        acceleration_structure_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::HOST,
            &build_info,
            &[1], // One instance (the cube BLAS)
            &mut size_info,
        );

        let (tlas_buffer, tlas_memory) = BuffersState::create_buffer(
            &init_state.instance,
            &init_state.device,
            init_state.physical_device,
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let tlas = acceleration_structure_loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(tlas_buffer)
                .size(size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
            None,
        )?;

        let (scratch_buffer, scratch_memory) = BuffersState::create_buffer(
            &init_state.instance,
            &init_state.device,
            init_state.physical_device,
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let scratch_address = pipeline_state
            .buffer_device_address_loader
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer),
            );

        let command_buffer = init_state.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(init_state.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];

        init_state.device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let build_info =
            build_info
                .dst_acceleration_structure(tlas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_address,
                });

        acceleration_structure_loader.cmd_build_acceleration_structures(
            command_buffer,
            &[build_info],
            &[&[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)]],
        );

        init_state.device.end_command_buffer(command_buffer)?;
        init_state.device.queue_submit(
            init_state.queues.graphics,
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            vk::Fence::null(),
        )?;
        init_state
            .device
            .queue_wait_idle(init_state.queues.graphics)?;
        init_state
            .device
            .free_command_buffers(init_state.command_pool, &[command_buffer]);

        init_state.device.destroy_buffer(scratch_buffer, None);
        init_state.device.free_memory(scratch_memory, None);
        init_state.device.destroy_buffer(instances_buffer, None);
        init_state.device.free_memory(instances_memory, None);

        Ok((tlas, tlas_buffer, tlas_memory))
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
        tlas: vk::AccelerationStructureKHR,
        output_image_view: &vk::ImageView,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        let descriptor_sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize]),
        )?;

        for (frame, &descriptor_set) in descriptor_sets.iter().enumerate() {
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .push_next(
                            &mut vk::WriteDescriptorSetAccelerationStructureKHR::default()
                                .acceleration_structures(&[tlas]),
                        ),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .image_info(&[vk::DescriptorImageInfo::default()
                            .image_view(*output_image_view)
                            .image_layout(vk::ImageLayout::GENERAL)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(uniform_buffers[frame])
                            .offset(0)
                            .range(mem::size_of::<CameraGpu>() as u64)]),
                ],
                &[],
            );
        }

        Ok(descriptor_sets)
    }

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            init_state
                .device
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .unwrap();
            init_state
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

#[derive(Resource)]
pub struct CommandSyncState {
    command_buffers: Vec<vk::CommandBuffer>,
    sync_objects: SyncObjects,
    current_frame: u8,
}

impl CommandSyncState {
    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let command_buffers =
                Self::create_command_buffers(&init_state.device, init_state.command_pool)?;

            let sync_objects = SyncObjects::new(&init_state.device)?;

            Ok(Self {
                command_buffers,
                sync_objects,
                current_frame: 0,
            })
        }
    }

    pub fn draw_frame(
        &mut self,
        init_state: &InitState,
        swapchain_render_state: &mut SwapchainRenderState,
        pipeline_state: &PipelineState,
        buffers_state: &mut BuffersState,
        acceleration_structure_state: &AccelerationStructureState,
        window_size: Vec2,
        camera_gpu: CameraGpu,
    ) -> VkResult<()> {
        unsafe {
            self.update_uniform_buffers(buffers_state, camera_gpu)?;

            init_state.device.wait_for_fences(
                &[self.sync_objects.in_flight_fences[self.current_frame as usize]],
                true,
                u64::MAX,
            )?;

            let (image_index, _suboptimal) = match swapchain_render_state.loader.acquire_next_image(
                swapchain_render_state.swapchain,
                u64::MAX,
                self.sync_objects.image_available_semaphores[self.current_frame as usize],
                vk::Fence::null(),
            ) {
                Ok(i) => i,
                Err(vk::Result::SUBOPTIMAL_KHR) => return Ok(()),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    swapchain_render_state.recreate_swapchain(init_state, window_size)?;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            init_state
                .device
                .reset_fences(&[self.sync_objects.in_flight_fences[self.current_frame as usize]])?;

            init_state.device.reset_command_buffer(
                self.command_buffers[self.current_frame as usize],
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.record_command_buffer(
                init_state,
                swapchain_render_state,
                pipeline_state,
                buffers_state,
                acceleration_structure_state,
                self.command_buffers[self.current_frame as usize],
                image_index,
            )?;

            let wait_semaphores =
                &[self.sync_objects.image_available_semaphores[self.current_frame as usize]];
            let signal_semaphores =
                &[self.sync_objects.render_finished_semaphores[self.current_frame as usize]];

            init_state.device.queue_submit(
                init_state.queues.graphics,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[self.command_buffers[self.current_frame as usize]])
                    .signal_semaphores(signal_semaphores)],
                self.sync_objects.in_flight_fences[self.current_frame as usize],
            )?;

            match swapchain_render_state.loader.queue_present(
                init_state.queues.present,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(signal_semaphores)
                    .swapchains(&[swapchain_render_state.swapchain])
                    .image_indices(&[image_index]),
            ) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    swapchain_render_state.recreate_swapchain(init_state, window_size)?;
                }
                Err(e) => return Err(e),
            };
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

            Ok(())
        }
    }

    unsafe fn update_uniform_buffers(
        &mut self,
        buffers_state: &mut BuffersState,
        camera_gpu: CameraGpu,
    ) -> VkResult<()> {
        ptr::copy_nonoverlapping(
            camera_gpu.to_bytes().as_ptr(),
            (*buffers_state.uniform_buffers_mapped[self.current_frame as usize]).as_mut_ptr(),
            UNIFORM_BUFFER_SIZE,
        );
        Ok(())
    }

    unsafe fn record_command_buffer(
        &mut self,
        init_state: &InitState,
        swapchain_render_state: &SwapchainRenderState,
        pipeline_state: &PipelineState,
        buffers_state: &BuffersState,
        acceleration_structure_state: &AccelerationStructureState,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
    ) -> VkResult<()> {
        init_state
            .device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

        // init_state.device.cmd_begin_render_pass(
        //     command_buffer,
        //     &vk::RenderPassBeginInfo::default()
        //         .render_pass(swapchain_render_state.render_pass)
        //         .framebuffer(swapchain_render_state.framebuffers[image_index as usize])
        //         .render_area(vk::Rect2D::default().extent(swapchain_render_state.extent))
        //         .clear_values(&[vk::ClearValue {
        //             color: vk::ClearColorValue {
        //                 float32: [0.0, 0.0, 0.0, 1.0],
        //             },
        //         }]),
        //     vk::SubpassContents::INLINE,
        // );

        init_state.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(swapchain_render_state.images[image_index as usize])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )],
        );

        init_state.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_state.pipeline,
        );

        init_state.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_state.pipeline_layout,
            0,
            &[acceleration_structure_state.descriptor_sets[self.current_frame as usize]],
            &[],
        );

        pipeline_state.ray_tracing_loader.cmd_trace_rays(
            command_buffer,
            &pipeline_state.shader_binding_table.raygen_region,
            &pipeline_state.shader_binding_table.miss_region,
            &pipeline_state.shader_binding_table.hit_region,
            &vk::StridedDeviceAddressRegionKHR::default(),
            swapchain_render_state.extent.width,
            swapchain_render_state.extent.height,
            1,
        );

        init_state.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::NONE)
                .image(swapchain_render_state.images[image_index as usize])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )],
        );

        init_state.device.end_command_buffer(command_buffer)?;

        Ok(())
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

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
                init_state
                    .device
                    .destroy_semaphore(self.sync_objects.image_available_semaphores[i], None);
                init_state
                    .device
                    .destroy_semaphore(self.sync_objects.render_finished_semaphores[i], None);
                init_state
                    .device
                    .destroy_fence(self.sync_objects.in_flight_fences[i], None);
            }
        }
    }
}

const MAX_FRAMES_IN_FLIGHT: u8 = 2;

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
    transfer_family: u32,
}

impl QueueFamilyIndices {
    pub const fn all(&self) -> [u32; 3] {
        [
            self.graphics_family,
            self.present_family,
            self.transfer_family,
        ]
    }
}

struct Queues {
    graphics: vk::Queue,
    transfer: vk::Queue,
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

        let transfer_family = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, properties)| {
                if properties.queue_flags.contains(vk::QueueFlags::TRANSFER) {
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
            transfer_family,
            present_family,
        })
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
