use std::{ptr, slice};

use ash::{prelude::VkResult, vk};

use crate::init_state::Queue;

pub struct Buffer<'a> {
    size: u64,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<&'a mut [u8]>,
}

impl<'a> Buffer<'a> {
    pub const fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub const fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub const fn mapped_mut(&mut self) -> &mut Option<&'a mut [u8]> {
        &mut self.mapped
    }

    pub fn create(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> VkResult<Self> {
        unsafe {
            let handle = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?; // TODO: check `EXCLUSIVE`

            let memory_requirements = device.get_buffer_memory_requirements(handle);

            let mut memory_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(memory_requirements.size)
                .memory_type_index(
                    Self::find_memory_type(
                        instance,
                        physical_device,
                        memory_requirements.memory_type_bits,
                        properties,
                    )?
                    .0,
                );

            let mut memory_allocate_flags = vk::MemoryAllocateFlagsInfo::default();
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                memory_allocate_flags.flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
                memory_allocate_info = memory_allocate_info.push_next(&mut memory_allocate_flags);
            }

            let memory = device.allocate_memory(&memory_allocate_info, None)?;

            device.bind_buffer_memory(handle, memory, 0)?;

            Ok(Self {
                size,
                handle,
                memory,
                mapped: None,
            })
        }
    }

    pub fn create_from_bytes_with_staging(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        command_fence: vk::Fence,
        transfer_queue: &Queue,
        bytes: &[u8],
        buffer_usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        unsafe {
            let size = bytes.len() as u64;
            let mut staging_buffer = Self::create(
                instance,
                device,
                physical_device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            staging_buffer.map_memory(device, 0, vk::MemoryMapFlags::empty())?;
            staging_buffer.write(bytes);
            staging_buffer.unmap_memory(device)?;

            let buffer = Self::create(
                instance,
                device,
                physical_device,
                size,
                vk::BufferUsageFlags::TRANSFER_DST | buffer_usage,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            Self::copy_handles(
                device,
                command_fence,
                transfer_queue,
                staging_buffer.handle(),
                buffer.handle(),
                size,
            )?;
            staging_buffer.cleanup(device);

            Ok(buffer)
        }
    }

    unsafe fn copy_handles(
        device: &ash::Device,
        command_fence: vk::Fence,
        transfer_queue: &Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) -> VkResult<()> {
        let command_buffer =
            Self::begin_single_time_commands(device, transfer_queue.command_pool().unwrap())?;
        device.cmd_copy_buffer(
            command_buffer,
            src,
            dst,
            &[vk::BufferCopy::default().size(size)],
        );
        Self::end_single_time_commands(device, command_buffer, command_fence, transfer_queue)?;
        Ok(())
    }

    pub unsafe fn begin_single_time_commands(
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

    pub unsafe fn end_single_time_commands(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        command_fence: vk::Fence,
        queue: &Queue,
    ) -> VkResult<()> {
        device.end_command_buffer(command_buffer)?;

        device.reset_fences(&[command_fence])?;
        device.queue_submit(
            queue.primary_handle().unwrap(),
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            command_fence,
        )?;
        device.wait_for_fences(&[command_fence], true, u64::MAX)?;
        device.free_command_buffers(queue.command_pool().unwrap(), &[command_buffer]);

        Ok(())
    }

    pub fn map_memory(
        &mut self,
        device: &ash::Device,
        offset: u64,
        flags: vk::MemoryMapFlags,
    ) -> VkResult<()> {
        debug_assert!(self.mapped.is_none(), "Memory already mapped!");
        unsafe {
            self.mapped = Some(slice::from_raw_parts_mut(
                device.map_memory(self.memory, offset, self.size, flags)? as *mut u8,
                self.size as usize,
            ));
            Ok(())
        }
    }

    pub fn unmap_memory(&mut self, device: &ash::Device) -> VkResult<()> {
        debug_assert!(self.mapped.is_some(), "Memory not mapped!");
        unsafe {
            device.unmap_memory(self.memory);
            self.mapped = None;
            Ok(())
        }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        match &mut self.mapped {
            None => panic!("Memory not mapped!"),
            Some(mapped) => unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.as_mut_ptr(), bytes.len());
            },
        }
    }

    pub fn find_memory_type(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> VkResult<(u32, vk::MemoryType)> {
        unsafe {
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
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            if self.mapped.is_some() {
                device.unmap_memory(self.memory);
            }
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.handle, None);
        }
    }
}
