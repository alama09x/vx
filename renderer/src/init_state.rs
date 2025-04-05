use std::{
    borrow::Cow,
    collections::HashSet,
    error::Error,
    ffi::{c_void, CStr, CString},
    os::raw,
};

use ash::{
    ext::debug_utils,
    khr::{self, surface},
    prelude::VkResult,
    vk,
};
use bevy_ecs::system::Resource;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

#[derive(Resource)]
pub struct InitState {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queues: Queues,
}

impl InitState {
    const ENGINE_NAME: &str = "VX Engine";
    const ENGINE_VERSION: u32 = 0;
    const API_VERSION: u32 = vk::make_api_version(1, 4, 0, 0);

    const LAYER_NAMES: &[&CStr] = &[c"VK_LAYER_KHRONOS_validation"];
    const DEVICE_EXTENSION_NAMES: &[&CStr] = &[
        khr::swapchain::NAME,
        khr::ray_tracing_pipeline::NAME,
        khr::acceleration_structure::NAME,
        khr::deferred_host_operations::NAME,
        khr::buffer_device_address::NAME,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME,
    ];

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn surface_loader(&self) -> &surface::Instance {
        &self.surface_loader
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn queues(&self) -> &Queues {
        &self.queues
    }

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

            println!("Before physical device");
            let (physical_device, mut queues) =
                Self::pick_physical_device(&instance, &surface_loader, surface)?;
            println!("After physical device");

            let device = Self::create_logical_device(&instance, physical_device, &queues)?;
            Self::initialize_queues(&device, &mut queues)?;
            queues.initialize_fence(&device)?;
            println!("Queue indices: {:?}", queues.indices());

            Ok(Self {
                _entry: entry,
                instance,
                debug_utils_loader,
                debug_messenger,
                surface_loader,
                surface,
                physical_device,
                device,
                queues,
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
    ) -> Result<(vk::PhysicalDevice, Queues), Box<dyn Error>> {
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
    ) -> VkResult<HashSet<String>> {
        let available_extensions =
            instance.enumerate_device_extension_properties(physical_device)?;
        let required_extensions: HashSet<_> = Self::DEVICE_EXTENSION_NAMES
            .iter()
            .map(|e| e.to_string_lossy().into_owned())
            .collect();

        let mut missing_extensions = required_extensions.clone();
        for ext in available_extensions.iter() {
            if let Ok(ext_name) = ext.extension_name_as_c_str() {
                missing_extensions.remove(&ext_name.to_string_lossy().into_owned());
            }
        }

        println!("Required extensions: {required_extensions:?}");
        println!("Missing extensions: {missing_extensions:?}");
        Ok(missing_extensions)
    }

    /// Returns `Some(Queue)` if the device is suitable
    unsafe fn device_is_suitable(
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Option<Queues>> {
        let queues =
            Queues::new_with_family_indices(instance, physical_device, surface_loader, surface)?;
        let missing_extensions = Self::check_device_extension_support(instance, physical_device)?;
        let extensions_supported = missing_extensions.is_empty();

        let swapchain_adequate = {
            let swapchain_support =
                SwapchainSupportDetails::new(physical_device, surface_loader, surface)?;
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        };
        let supported_features = instance.get_physical_device_features(physical_device);

        if extensions_supported && swapchain_adequate && supported_features.sampler_anisotropy != 0
        {
            Ok(Some(queues))
        } else {
            Ok(None)
        }
    }

    unsafe fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queues: &Queues,
    ) -> VkResult<ash::Device> {
        let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);

        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true); // Already present, keep this
        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        // Chain the feature structs
        vulkan11_features.p_next = &mut buffer_device_address_features as *mut _ as *mut c_void;
        buffer_device_address_features.p_next =
            &mut ray_tracing_pipeline_features as *mut _ as *mut c_void;
        ray_tracing_pipeline_features.p_next =
            &mut acceleration_structure_features as *mut _ as *mut c_void;

        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(
                    // Unique queue family indices
                    &queues
                        .indices()
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
                .push_next(&mut vulkan11_features)
                .enabled_features(&vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true)),
            None,
        )?;
        Ok(device)
    }

    unsafe fn initialize_queues(device: &ash::Device, queues: &mut Queues) -> VkResult<()> {
        unsafe {
            *queues.graphics.primary_handle_mut() =
                Some(device.get_device_queue(queues.graphics.family_index, 0));
            *queues.transfer.primary_handle_mut() =
                Some(device.get_device_queue(queues.transfer.family_index, 0));
            *queues.present.primary_handle_mut() =
                Some(device.get_device_queue(queues.present.family_index, 0));

            *queues.graphics.command_pool_mut() = Some(Self::create_command_pool(
                device,
                queues.graphics.family_index,
            )?);
            *queues.transfer.command_pool_mut() = Some(Self::create_command_pool(
                device,
                queues.transfer.family_index,
            )?);
            *queues.present.command_pool_mut() = Some(Self::create_command_pool(
                device,
                queues.present.family_index,
            )?);

            Ok(())
        }
    }

    unsafe fn create_command_pool(
        device: &ash::Device,
        family_index: u32,
    ) -> VkResult<vk::CommandPool> {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family_index),
            None,
        )
    }
}

impl Drop for InitState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device
                .destroy_fence(self.queues.command_fence().unwrap(), None);
            for command_pool in self.queues.command_pools() {
                self.device
                    .destroy_command_pool(command_pool.unwrap(), None);
            }

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct Queue {
    family_index: u32,
    primary_handle: Option<vk::Queue>,
    command_pool: Option<vk::CommandPool>,
}

impl Queue {
    pub fn new_with_family_index(family_index: u32) -> Self {
        Self {
            family_index,
            primary_handle: None,
            command_pool: None,
        }
    }

    pub const fn family_index(&self) -> u32 {
        self.family_index
    }

    pub const fn primary_handle(&self) -> Option<vk::Queue> {
        self.primary_handle
    }

    pub const fn primary_handle_mut(&mut self) -> &mut Option<vk::Queue> {
        &mut self.primary_handle
    }

    pub const fn command_pool(&self) -> Option<vk::CommandPool> {
        self.command_pool
    }

    pub const fn command_pool_mut(&mut self) -> &mut Option<vk::CommandPool> {
        &mut self.command_pool
    }
}

pub struct Queues {
    pub graphics: Queue,
    pub transfer: Queue,
    pub present: Queue,
    command_fence: Option<vk::Fence>,
}

impl Queues {
    pub const COUNT: u8 = 3;

    pub const fn graphics(&self) -> &Queue {
        &self.graphics
    }

    pub const fn transfer(&self) -> &Queue {
        &self.transfer
    }

    pub const fn present(&self) -> &Queue {
        &self.present
    }

    pub const fn command_fence(&self) -> Option<vk::Fence> {
        self.command_fence
    }

    pub const fn indices(&self) -> [u32; Self::COUNT as usize] {
        [
            self.graphics.family_index(),
            self.present.family_index(),
            self.transfer.family_index(),
        ]
    }

    pub const fn command_pools(&self) -> [Option<vk::CommandPool>; Self::COUNT as usize] {
        [
            self.graphics.command_pool(),
            self.transfer.command_pool(),
            self.present.command_pool(),
        ]
    }

    pub fn new_with_family_indices(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Self> {
        unsafe {
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);

            let graphics_family_index = queue_families
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

            let transfer_family_index = queue_families
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
                graphics: Queue::new_with_family_index(graphics_family_index),
                transfer: Queue::new_with_family_index(transfer_family_index),
                present: Queue::new_with_family_index(present_family),
                command_fence: None,
            })
        }
    }

    pub fn initialize_fence(&mut self, device: &ash::Device) -> VkResult<()> {
        unsafe {
            self.command_fence = Some(device.create_fence(&vk::FenceCreateInfo::default(), None)?);
            Ok(())
        }
    }
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> VkResult<Self> {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)?;

            let formats =
                surface_loader.get_physical_device_surface_formats(physical_device, surface)?;

            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)?;

            Ok(Self {
                capabilities,
                formats,
                present_modes,
            })
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
