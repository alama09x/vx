use std::{
    error::Error,
    fs::File,
    io::{self, Read},
    path::Path,
};

use ash::{
    khr::{buffer_device_address, ray_tracing_pipeline},
    prelude::VkResult,
    vk,
};
use bevy_ecs::system::Resource;

use crate::{buffer::Buffer, init_state::InitState};

#[derive(Resource)]
pub struct PipelineState<'a> {
    ray_tracing_loader: ray_tracing_pipeline::Device,
    buffer_device_address_loader: buffer_device_address::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_binding_table: ShaderBindingTable<'a>,
}

impl<'a> PipelineState<'a> {
    pub const fn ray_tracing_loader(&self) -> &ray_tracing_pipeline::Device {
        &self.ray_tracing_loader
    }

    pub const fn buffer_device_address_loader(&self) -> &buffer_device_address::Device {
        &self.buffer_device_address_loader
    }

    pub const fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub const fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub const fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub const fn shader_binding_table(&self) -> &ShaderBindingTable {
        &self.shader_binding_table
    }

    pub const fn shader_binding_table_mut(&'a mut self) -> &'a mut ShaderBindingTable<'a> {
        &mut self.shader_binding_table
    }

    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let ray_tracing_loader =
                ray_tracing_pipeline::Device::new(init_state.instance(), init_state.device());
            let buffer_device_address_loader =
                buffer_device_address::Device::new(init_state.instance(), init_state.device());

            let descriptor_set_layout = Self::create_descriptor_set_layout(init_state.device())?;

            let (pipeline_layout, pipeline) = Self::create_pipeline(
                init_state.device(),
                &ray_tracing_loader,
                descriptor_set_layout,
            )?;

            let shader_binding_table = Self::create_shader_binding_table(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
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
    ) -> Result<ShaderBindingTable<'a>, Box<dyn Error>> {
        let mut rt_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        instance.get_physical_device_properties2(
            physical_device,
            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut rt_properties),
        );

        let handle_size = rt_properties.shader_group_handle_size as vk::DeviceSize;
        let group_count = 3;

        let group_alignment = rt_properties
            .shader_group_handle_alignment
            .max(rt_properties.shader_group_base_alignment)
            .max(64) as vk::DeviceSize;

        let total_size = group_alignment * group_count;

        if handle_size == 0 || total_size == 0 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Shader group handle size is 0, properties query failed",
            )));
        }

        let mut buffer = Buffer::create(
            instance,
            device,
            physical_device,
            total_size,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        buffer.map_memory(device, 0, vk::MemoryMapFlags::empty())?;

        let handles = rt_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count as u32,
            (handle_size * group_count) as usize,
        )?;
        let mapped = buffer.mapped_mut().as_mut().unwrap();
        mapped[0..handle_size as usize].copy_from_slice(&handles[0..handle_size as usize]); // Raygen at 0
        mapped[group_alignment as usize..(group_alignment + handle_size) as usize]
            .copy_from_slice(&handles[handle_size as usize..(handle_size * 2) as usize]); // Miss at 64
        mapped[(group_alignment * 2) as usize..(group_alignment * 2 + handle_size) as usize]
            .copy_from_slice(&handles[(handle_size * 2) as usize..]); // Hit at 128
        buffer.unmap_memory(device)?;

        let buffer_address = bda_loader.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(buffer.handle()),
        );

        let aligned_buffer_address =
            (buffer_address + group_alignment - 1) & !(group_alignment - 1);

        let region_size = handle_size;
        Ok(ShaderBindingTable {
            buffer,
            raygen_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(aligned_buffer_address)
                .stride(region_size)
                .size(region_size),
            miss_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(aligned_buffer_address + group_alignment)
                .stride(region_size)
                .size(region_size),
            hit_region: vk::StridedDeviceAddressRegionKHR::default()
                .device_address(aligned_buffer_address + group_alignment * 2)
                .stride(region_size)
                .size(region_size),
        })
    }

    pub fn cleanup(&mut self, init_state: &InitState) {
        unsafe {
            self.shader_binding_table
                .buffer
                .cleanup(init_state.device());

            init_state.device().destroy_pipeline(self.pipeline, None);
            init_state
                .device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            init_state
                .device()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct ShaderBindingTable<'a> {
    buffer: Buffer<'a>,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}
