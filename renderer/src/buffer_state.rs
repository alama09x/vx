use std::error::Error;

use ash::{prelude::VkResult, vk};
use bevy_ecs::system::Resource;

use crate::{
    buffer::Buffer,
    init_state::{InitState, Queue},
    INDICES, MAX_FRAMES_IN_FLIGHT, UNIFORM_BUFFER_SIZE, VERTICES,
};

#[derive(Resource)]
pub struct BufferState<'a> {
    vertex_buffer: Buffer<'a>,
    index_buffer: Buffer<'a>,
    uniform_buffers: Vec<Buffer<'a>>,
}

impl<'a> BufferState<'a> {
    pub fn vertex_buffer(&self) -> &Buffer<'a> {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &Buffer<'a> {
        &self.index_buffer
    }

    pub fn uniform_buffers(&self) -> &[Buffer<'a>] {
        &self.uniform_buffers
    }

    pub fn uniform_buffers_mut(&mut self) -> &mut [Buffer<'a>] {
        &mut self.uniform_buffers
    }

    pub fn new(init_state: &InitState) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let vertex_buffer = Self::create_vertex_buffer(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
                init_state.queues().command_fence().unwrap(),
                init_state.queues().transfer(),
            )?;

            let index_buffer = Self::create_index_buffer(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
                init_state.queues().command_fence().unwrap(),
                init_state.queues().transfer(),
            )?;

            let uniform_buffers = Self::create_uniform_buffers(
                init_state.instance(),
                init_state.device(),
                init_state.physical_device(),
                MAX_FRAMES_IN_FLIGHT,
            )?;

            Ok(Self {
                vertex_buffer,
                index_buffer,
                uniform_buffers,
            })
        }
    }

    unsafe fn create_vertex_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_fence: vk::Fence,
        transfer_queue: &Queue,
    ) -> VkResult<Buffer<'a>> {
        Buffer::create_from_bytes_with_staging(
            instance,
            device,
            physical_device,
            command_fence,
            transfer_queue,
            bytemuck::cast_slice(&VERTICES),
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )
    }

    unsafe fn create_index_buffer(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_fence: vk::Fence,
        transfer_queue: &Queue,
    ) -> VkResult<Buffer<'a>> {
        Buffer::create_from_bytes_with_staging(
            instance,
            device,
            physical_device,
            command_fence,
            transfer_queue,
            bytemuck::cast_slice(&INDICES),
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )
    }

    unsafe fn create_uniform_buffers(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        frames: u8,
    ) -> VkResult<Vec<Buffer<'a>>> {
        let buffer_size = UNIFORM_BUFFER_SIZE;

        let mut buffers = Vec::with_capacity(frames as usize);

        for _ in 0..frames as usize {
            let mut buffer = Buffer::create(
                instance,
                device,
                physical_device,
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | { vk::MemoryPropertyFlags::HOST_COHERENT },
            )?;
            buffer.map_memory(device, 0, vk::MemoryMapFlags::empty())?;
            buffers.push(buffer);
        }

        Ok(buffers)
    }

    pub fn cleanup(&mut self, init_state: &InitState) {
        self.vertex_buffer.cleanup(init_state.device());
        self.index_buffer.cleanup(init_state.device());
        for uniform_buffer in &mut self.uniform_buffers {
            uniform_buffer.cleanup(init_state.device());
        }
    }
}
