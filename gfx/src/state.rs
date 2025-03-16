use std::{
    error::Error,
    ffi::{CStr, CString},
};

use ash::{
    ext::debug_utils,
    vk::{self},
    Entry,
};
use winit::{raw_window_handle::HasDisplayHandle, window::Window};

pub struct VxState {
    app_name: &'static str,
    app_version: u32,
    entry: ash::Entry,
    window: Option<Window>,
    instance: ash::Instance,
}

impl VxState {
    const ENGINE_NAME: &str = "VX Engine";
    const ENGINE_VERSION: u32 = 0;
    const API_VERSION: u32 = vk::API_VERSION_1_3;

    const LAYER_NAMES: &[&CStr] = &[c"VK_LAYER_KHRONOS_validation"];

    pub fn new(
        app_name: &'static str,
        app_version: u32,
        window: &Window,
    ) -> Result<Self, Box<dyn Error>> {
        let entry = unsafe { Entry::load() }?;
        let instance = Self::create_instance(&entry, app_name, app_version, window)?;

        Ok(Self {
            app_name,
            app_version,
            entry,
            window: None,
            instance,
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        app_name: &str,
        app_version: u32,
        window: &Window,
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
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
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

        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        Ok(instance)
    }
}

impl Drop for VxState {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
