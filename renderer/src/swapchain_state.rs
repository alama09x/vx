use std::{collections::HashSet, error::Error};

use ash::{
    khr::{surface, swapchain},
    prelude::VkResult,
    vk,
};
use bevy_ecs::system::Resource;
use glam::Vec2;

use crate::{
    acceleration_structure_state::AccelerationStructureState,
    buffer::Buffer,
    buffer_state::BufferState,
    init_state::{InitState, Queue, Queues, SwapchainSupportDetails},
    MAX_FRAMES_IN_FLIGHT,
};

#[derive(Resource)]
pub struct SwapchainState {
    loader: swapchain::Device,
    image_format: vk::Format,
    extent: vk::Extent2D,

    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    output_images: Vec<vk::Image>,
    output_image_memories: Vec<vk::DeviceMemory>,
    output_image_views: Vec<vk::ImageView>,
}

impl SwapchainState {
    pub const fn extent(&self) -> &vk::Extent2D {
        &self.extent
    }

    pub const fn output_images(&self) -> &Vec<vk::Image> {
        &self.output_images
    }

    pub const fn output_image_views(&self) -> &Vec<vk::ImageView> {
        &self.output_image_views
    }

    pub const fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub const fn images(&self) -> &Vec<vk::Image> {
        &self.images
    }

    pub const fn image_views(&self) -> &Vec<vk::ImageView> {
        &self.image_views
    }

    pub const fn loader(&self) -> &swapchain::Device {
        &self.loader
    }

    pub fn new(init_state: &InitState, window_size: Vec2) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let loader = swapchain::Device::new(init_state.instance(), init_state.device());

            let (swapchain, image_format, extent, images) = Self::create_swapchain(
                init_state.device(),
                init_state.physical_device(),
                init_state.surface_loader(),
                init_state.surface(),
                init_state.queues(),
                &loader,
                window_size,
            )?;

            let image_views = Self::create_image_views(init_state.device(), image_format, &images)?;

            let (output_images, output_image_memories) = Self::create_output_images(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
                init_state.queues().command_fence().unwrap(),
                init_state.queues().graphics(),
                extent,
            )?;

            let output_image_views =
                Self::create_image_views(init_state.device(), image_format, &output_images)?;

            Ok(Self {
                loader,
                image_format,
                extent,

                swapchain,
                images,
                image_views,

                output_images,
                output_image_memories,
                output_image_views,
            })
        }
    }

    pub fn recreate_swapchain(
        &mut self,
        init_state: &InitState,
        buffer_state: &BufferState,
        acceleration_structure_state: &mut AccelerationStructureState,
        window_size: Vec2,
    ) -> VkResult<()> {
        unsafe {
            init_state.device().device_wait_idle()?;
            if window_size.x == 0.0 || window_size.y == 0.0 {
                return Ok(());
            }

            self.cleanup_swapchain(init_state);
            (self.swapchain, self.image_format, self.extent, self.images) = Self::create_swapchain(
                init_state.device(),
                init_state.physical_device(),
                init_state.surface_loader(),
                init_state.surface(),
                init_state.queues(),
                &self.loader,
                window_size,
            )?;

            self.image_views =
                Self::create_image_views(init_state.device(), self.image_format, &self.images)?;

            (self.output_images, self.output_image_memories) = Self::create_output_images(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
                init_state.queues().command_fence().unwrap(),
                init_state.queues().graphics(),
                self.extent,
            )?;
            self.output_image_views = Self::create_image_views(
                init_state.device(),
                self.image_format,
                self.output_images(),
            )?;
            acceleration_structure_state.update_descriptor_sets(
                init_state.device(),
                buffer_state.uniform_buffers(),
                self.output_image_views(),
            );

            Ok(())
        }
    }

    fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<&vk::SurfaceFormatKHR> {
        formats.iter().find(|f| {
            f.format == vk::Format::R8G8B8A8_UNORM
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
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_loader: &surface::Instance,
        surface: vk::SurfaceKHR,
        queues: &Queues,
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

        let unique_indices: Vec<_> = queues
            .indices()
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
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
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

        let command_buffer =
            Buffer::begin_single_time_commands(device, queues.graphics().command_pool().unwrap())?;

        for image in &swapchain_images {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_access_mask(vk::AccessFlags::NONE)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .image(*image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );
        }

        Buffer::end_single_time_commands(
            device,
            command_buffer,
            queues.command_fence().unwrap(),
            queues.graphics(),
        )?;

        Ok((swapchain, surface_format.format, extent, swapchain_images))
    }

    unsafe fn create_image_view(
        device: &ash::Device,
        format: vk::Format,
        image: vk::Image,
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
            .map(|&image| Self::create_image_view(device, format, image))
            .collect()
    }

    unsafe fn cleanup_swapchain(&self, init_state: &InitState) {
        for &image_view in &self.image_views {
            init_state.device().destroy_image_view(image_view, None);
        }

        for i in 0..MAX_FRAMES_IN_FLIGHT as usize {
            init_state
                .device()
                .destroy_image_view(self.output_image_views[i], None);
            init_state
                .device()
                .destroy_image(self.output_images[i], None);
            init_state
                .device()
                .free_memory(self.output_image_memories[i], None);
        }

        self.loader.destroy_swapchain(self.swapchain, None);
    }

    pub fn cleanup(&self, init_state: &InitState) {
        unsafe {
            self.cleanup_swapchain(init_state);
        }
    }

    fn create_output_images(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_fence: vk::Fence,
        queue: &Queue,
        extent: vk::Extent2D,
    ) -> VkResult<(Vec<vk::Image>, Vec<vk::DeviceMemory>)> {
        unsafe {
            let mut images = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
            let mut memories = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let image = device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM) // TODO: check if supported on device
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
                let (memory_type_index, _) = Buffer::find_memory_type(
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

                let command_buffer =
                    Buffer::begin_single_time_commands(device, queue.command_pool().unwrap())?;

                device.cmd_pipeline_barrier(
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
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )],
                );

                Buffer::end_single_time_commands(device, command_buffer, command_fence, queue)?;
                images.push(image);
                memories.push(memory);
            }
            Ok((images, memories))
        }
    }
}
