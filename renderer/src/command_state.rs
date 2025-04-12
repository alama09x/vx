use std::error::Error;

use ash::{prelude::VkResult, vk};
use bevy_ecs::system::Resource;
use data::{camera::CameraGpu, IntoBytes};

use glam::Vec2;

use crate::{
    acceleration_structure_state::AccelerationStructureState, buffer_state::BufferState,
    init_state::InitState, pipeline_state::PipelineState, swapchain_state::SwapchainState,
};

#[derive(Resource)]
pub struct CommandState {
    command_buffers: Vec<vk::CommandBuffer>,
    sync_objects: SyncObjects,
}

impl CommandState {
    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let command_buffers = Self::create_command_buffers(
                init_state.device(),
                init_state.queues().graphics().command_pool().unwrap(),
            )?;

            let sync_objects = SyncObjects::new(init_state.device())?;

            Ok(Self {
                command_buffers,
                sync_objects,
            })
        }
    }

    pub fn draw_frame(
        &mut self,
        init_state: &InitState,
        swapchain_state: &mut SwapchainState,
        pipeline_state: &PipelineState,
        buffer_state: &mut BufferState,
        acceleration_structure_state: &mut AccelerationStructureState,
        window_size: Vec2,
        camera_gpu: CameraGpu,
        current_frame: u8,
    ) -> VkResult<()> {
        unsafe {
            self.update_uniform_buffers(buffer_state, camera_gpu, current_frame)?;

            init_state.device().wait_for_fences(
                &[self.sync_objects.in_flight_fences[current_frame as usize]],
                true,
                u64::MAX,
            )?;

            let (image_index, _suboptimal) = match swapchain_state.loader().acquire_next_image(
                swapchain_state.swapchain(),
                u64::MAX,
                self.sync_objects.image_available_semaphores[current_frame as usize],
                vk::Fence::null(),
            ) {
                Ok(i) => i,
                Err(vk::Result::SUBOPTIMAL_KHR) => return Ok(()),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    swapchain_state.recreate_swapchain(
                        init_state,
                        buffer_state,
                        acceleration_structure_state,
                        window_size,
                    )?;
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            init_state
                .device()
                .reset_fences(&[self.sync_objects.in_flight_fences[current_frame as usize]])?;

            init_state.device().reset_command_buffer(
                self.command_buffers[current_frame as usize],
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.record_command_buffer(
                init_state,
                swapchain_state,
                pipeline_state,
                acceleration_structure_state,
                self.command_buffers[current_frame as usize],
                image_index,
                current_frame,
            )?;

            let wait_semaphores =
                &[self.sync_objects.image_available_semaphores[current_frame as usize]];
            let signal_semaphores =
                &[self.sync_objects.render_finished_semaphores[current_frame as usize]];

            init_state.device().queue_submit(
                init_state.queues().graphics().primary_handle().unwrap(),
                &[vk::SubmitInfo::default()
                    .wait_semaphores(wait_semaphores)
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&[self.command_buffers[current_frame as usize]])
                    .signal_semaphores(signal_semaphores)],
                self.sync_objects.in_flight_fences[current_frame as usize],
            )?;

            match swapchain_state.loader().queue_present(
                init_state.queues().present().primary_handle().unwrap(),
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(signal_semaphores)
                    .swapchains(&[swapchain_state.swapchain()])
                    .image_indices(&[image_index]),
            ) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    swapchain_state.recreate_swapchain(
                        init_state,
                        buffer_state,
                        acceleration_structure_state,
                        window_size,
                    )?;
                }
                Err(e) => return Err(e),
            };
            Ok(())
        }
    }

    unsafe fn update_uniform_buffers(
        &mut self,
        buffer_state: &mut BufferState,
        camera_gpu: CameraGpu,
        current_frame: u8,
    ) -> VkResult<()> {
        buffer_state.uniform_buffers_mut()[current_frame as usize].write(camera_gpu.to_bytes());
        // let mapped = buffer_state.uniform_buffers()[current_frame as usize]
        //     .mapped()
        //     .as_ref()
        //     .unwrap();
        // let camera_data: &CameraGpu = bytemuck::from_bytes(mapped);
        // println!("GPU View Inverse:\n{:?}", camera_data.view_inverse);
        // println!("GPU Proj Inverse:\n{:?}", camera_data.proj_inverse);
        Ok(())
    }

    unsafe fn record_command_buffer(
        &mut self,
        init_state: &InitState,
        swapchain_state: &SwapchainState,
        pipeline_state: &PipelineState,
        acceleration_structure_state: &AccelerationStructureState,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
        current_frame: u8,
    ) -> VkResult<()> {
        init_state
            .device()
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

        // Transition swapchain image from PRESENT_SRC_KHR to TRANSFER_DST_OPTIMAL
        init_state.device().cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .image(swapchain_state.images()[image_index as usize])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )],
        );

        // Ray tracing (output_image already in GENERAL from descriptor setup)
        init_state.device().cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_state.pipeline(),
        );

        init_state.device().cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline_state.pipeline_layout(),
            0,
            &[acceleration_structure_state.descriptor_sets()[current_frame as usize]],
            &[],
        );

        pipeline_state.ray_tracing_loader().cmd_trace_rays(
            command_buffer,
            &pipeline_state.shader_binding_table().raygen_region,
            &pipeline_state.shader_binding_table().miss_region,
            &pipeline_state.shader_binding_table().hit_region,
            &vk::StridedDeviceAddressRegionKHR::default(),
            swapchain_state.extent().width,
            swapchain_state.extent().height,
            1,
        );

        // Transition output_image to TRANSFER_SRC_OPTIMAL
        init_state.device().cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(swapchain_state.output_images()[current_frame as usize])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )],
        );

        // Blit from output_image to swapchain image
        init_state.device().cmd_blit_image(
            command_buffer,
            swapchain_state.output_images()[current_frame as usize],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_state.images()[image_index as usize],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: swapchain_state.extent().width as i32,
                        y: swapchain_state.extent().height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: swapchain_state.extent().width as i32,
                        y: swapchain_state.extent().height as i32,
                        z: 1,
                    },
                ])],
            vk::Filter::NEAREST,
        );

        // Transition swapchain to PRESENT_SRC_KHR and output_image back to GENERAL
        init_state.device().cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::NONE)
                    .image(swapchain_state.images()[image_index as usize])
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .image(swapchain_state.output_images()[current_frame as usize])
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
            ],
        );

        init_state.device().end_command_buffer(command_buffer)?;
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
                    .device()
                    .destroy_semaphore(self.sync_objects.image_available_semaphores[i], None);
                init_state
                    .device()
                    .destroy_semaphore(self.sync_objects.render_finished_semaphores[i], None);
                init_state
                    .device()
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
