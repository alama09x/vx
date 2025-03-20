#![warn(clippy::all)]

use std::{
    borrow::Cow,
    collections::HashSet,
    error::Error,
    ffi::{CStr, CString},
    fs::File,
    io::{self, Read},
    mem,
    os::raw,
    path::Path,
    slice, str,
};

use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    prelude::VkResult,
    vk,
};
use winit::{
    dpi::LogicalSize,
    error::OsError,
    event_loop::ActiveEventLoop,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowAttributes},
};

#[allow(dead_code)]
pub struct VxState {
    window: Window,
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
    // swapchain_framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

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
                Self::pick_physical_device(&instance, &surface_loader, &surface)?;

            let (device, queues) =
                Self::create_logical_device(&instance, &physical_device, &queue_family_indices)?;

            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let (swapchain, swapchain_images, swapchain_image_format, swapchain_extent) =
                Self::create_swapchain(
                    &physical_device,
                    &surface_loader,
                    &surface,
                    &swapchain_loader,
                    &window,
                    &queue_family_indices,
                )?;

            let swapchain_image_views =
                Self::create_image_views(&device, &swapchain_images, &swapchain_image_format)?;

            let render_pass = Self::create_render_pass(&device, &swapchain_image_format)?;

            let descriptor_set_layout = Self::create_descripitor_set_layout(&device)?;
            let (pipeline_layout, pipeline) =
                Self::create_graphics_pipeline(&device, &render_pass, &descriptor_set_layout)?;

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

                render_pass,
                descriptor_set_layout,
                pipeline_layout,
                pipeline,
            })
        }
    }

    fn create_window(
        title: &str,
        size: &LogicalSize<f32>,
        event_loop: &ActiveEventLoop,
    ) -> Result<Window, OsError> {
        let window = event_loop.create_window(
            WindowAttributes::default()
                .with_title(title)
                .with_inner_size(*size),
        )?;
        Ok(window)
    }

    unsafe fn create_instance(
        entry: &ash::Entry,
        app_name: &str,
        app_version: u32,
        event_loop: &ActiveEventLoop,
    ) -> Result<ash::Instance, Box<dyn Error>> {
        let app_name_raw = CString::new(app_name).unwrap();
        let engine_name_raw = CString::new(Self::ENGINE_NAME).unwrap();
        let layer_names_raw: Vec<_> = Self::LAYER_NAMES.iter().map(|name| name.as_ptr()).collect();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name_raw)
            .application_version(app_version)
            .engine_name(&engine_name_raw)
            .engine_version(Self::ENGINE_VERSION)
            .api_version(Self::API_VERSION);

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

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance = entry.create_instance(&create_info, None)?;
        Ok(instance)
    }

    unsafe fn create_debug_messenger(
        debug_utils_loader: &debug_utils::Instance,
    ) -> VkResult<vk::DebugUtilsMessengerEXT> {
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
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
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_callback = debug_utils_loader.create_debug_utils_messenger(&create_info, None)?;
        Ok(debug_callback)
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
        surface: &vk::SurfaceKHR,
    ) -> Result<(vk::PhysicalDevice, QueueFamilyIndices), Box<dyn Error>> {
        let physical_devices = instance.enumerate_physical_devices()?;
        physical_devices
            .iter()
            .find_map(|physical_device| {
                let indices =
                    Self::device_is_suitable(physical_device, instance, surface_loader, surface)
                        .ok()?;
                indices.map(|indices| (*physical_device, indices))
            })
            .ok_or(Box::new(vk::Result::ERROR_UNKNOWN))
    }

    unsafe fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> VkResult<bool> {
        let available_extensions =
            instance.enumerate_device_extension_properties(*physical_device)?;
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
        physical_device: &vk::PhysicalDevice,
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> VkResult<Option<QueueFamilyIndices>> {
        let indices = QueueFamilyIndices::new(instance, physical_device, surface_loader, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, physical_device)?;
        let swapchain_adequate = {
            let swapchain_support =
                SwapchainSupportDetails::new(physical_device, surface_loader, surface)?;
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        };
        let supported_features = instance.get_physical_device_features(*physical_device);

        if extensions_supported && swapchain_adequate && supported_features.sampler_anisotropy != 0
        {
            Ok(Some(indices))
        } else {
            Ok(None)
        }
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
    ) -> VkResult<(ash::Device, Queues)> {
        let device_extension_names_raw: Vec<_> = Self::DEVICE_EXTENSION_NAMES
            .iter()
            .map(|x| x.as_ptr())
            .collect();

        let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true);

        let priorities = [1.0];

        let unique_queue_create_infos: Vec<_> = queue_family_indices
            .all()
            .iter()
            .collect::<HashSet<_>>()
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(**index)
                    .queue_priorities(&priorities)
            })
            .collect();

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&unique_queue_create_infos)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device = instance.create_device(*physical_device, &create_info, None)?;
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
            let width = size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            );
            let height = size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            );

            vk::Extent2D { width, height }
        }
    }

    unsafe fn create_swapchain(
        physical_device: &vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: &vk::SurfaceKHR,
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

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
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
            .clipped(true);

        let swapchain = swapchain_loader.create_swapchain(&create_info, None)?;
        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain)?;

        Ok((swapchain, swapchain_images, surface_format.format, extent))
    }

    unsafe fn create_image_view(
        device: &ash::Device,
        image: &vk::Image,
        format: &vk::Format,
    ) -> VkResult<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(*format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = device.create_image_view(&create_info, None)?;
        Ok(image_view)
    }

    unsafe fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: &vk::Format,
    ) -> VkResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|image| Self::create_image_view(device, image, format))
            .collect()
    }

    unsafe fn cleanup_swapchain(
        device: &ash::Device,
        swapchain_loader: &swapchain::Device,
        swapchain: &vk::SwapchainKHR,
        swapchain_image_views: &[vk::ImageView],
    ) {
        for image_view in swapchain_image_views {
            device.destroy_image_view(*image_view, None);
        }
        swapchain_loader.destroy_swapchain(*swapchain, None);
    }

    unsafe fn create_descripitor_set_layout(
        device: &ash::Device,
    ) -> VkResult<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&[]);
        device.create_descriptor_set_layout(&create_info, None)
    }

    unsafe fn create_render_pass(
        device: &ash::Device,
        swapchain_image_format: &vk::Format,
    ) -> VkResult<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(*swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(slice::from_ref(&color_attachment_ref));

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::from_raw(0))
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(slice::from_ref(&color_attachment))
            .subpasses(slice::from_ref(&subpass))
            .dependencies(slice::from_ref(&dependency));

        let render_pass = device.create_render_pass(&create_info, None)?;
        Ok(render_pass)
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
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        device.create_shader_module(&create_info, None)
    }

    // unsafe fn create_descriptor_set_layout(device: &ash::Device) -> VkResult<vk::DescriptorSetLayout> {}
    unsafe fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: &vk::RenderPass,
        descriptor_set_layout: &vk::DescriptorSetLayout,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn Error>> {
        let vert_shader = Self::read_shader_code(Path::new("./bin/shader.vert.spv"))?;
        let frag_shader = Self::read_shader_code(Path::new("./bin/shader.frag.spv"))?;

        let vert_shader_module = Self::create_shader_module(device, &vert_shader)?;
        let frag_shader_module = Self::create_shader_module(device, &frag_shader)?;

        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(c"main");

        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(c"main");

        let shader_stages = [vert_shader_stage_create_info, frag_shader_stage_create_info];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(slice::from_ref(&Vertex::BINDING_DESCRIPTION))
            .vertex_attribute_descriptions(&Vertex::ATTRIBUTE_DESCRIPTIONS);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::R)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(slice::from_ref(&color_blend_attachment))
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(descriptor_set_layout));

        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_create_info, None)?;

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let pipelines = device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&pipeline_create_info),
                None,
            )
            .map_err(|_| vk::Result::ERROR_UNKNOWN)?;

        device.destroy_shader_module(frag_shader_module, None);
        device.destroy_shader_module(vert_shader_module, None);
        Ok((pipeline_layout, pipelines[0]))
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
        physical_device: &vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> VkResult<Self> {
        let capabilities =
            surface_loader.get_physical_device_surface_capabilities(*physical_device, *surface)?;

        let formats =
            surface_loader.get_physical_device_surface_formats(*physical_device, *surface)?;

        let present_modes =
            surface_loader.get_physical_device_surface_present_modes(*physical_device, *surface)?;

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
        physical_device: &vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> VkResult<Self> {
        let queue_families = instance.get_physical_device_queue_family_properties(*physical_device);

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
                    .get_physical_device_surface_support(*physical_device, index as u32, *surface)
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
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            Self::cleanup_swapchain(
                &self.device,
                &self.swapchain_loader,
                &self.swapchain,
                &self.swapchain_image_views,
            );
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
