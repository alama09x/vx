use std::{error::Error, mem, slice};

use ash::{khr::acceleration_structure, prelude::VkResult, vk};
use bevy_ecs::system::Resource;
use data::camera::CameraGpu;

use crate::{
    buffer::Buffer, buffer_state::BufferState, init_state::InitState,
    pipeline_state::PipelineState, swapchain_state::SwapchainState, Vertex, INDICES,
    MAX_FRAMES_IN_FLIGHT, VERTICES,
};

#[derive(Resource)]
pub struct AccelerationStructureState<'a> {
    loader: acceleration_structure::Device,
    fence: vk::Fence,
    blas: vk::AccelerationStructureKHR,
    blas_buffer: Buffer<'a>,
    tlas: vk::AccelerationStructureKHR,
    tlas_buffer: Buffer<'a>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl<'a> AccelerationStructureState<'a> {
    pub const fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }

    pub const fn descriptor_sets(&self) -> &Vec<vk::DescriptorSet> {
        &self.descriptor_sets
    }

    pub fn new(
        init_state: &InitState,
        swapchain_state: &SwapchainState,
        pipeline_state: &PipelineState,
        buffer_state: &BufferState,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let acceleration_structure_loader =
                acceleration_structure::Device::new(init_state.instance(), init_state.device());

            let fence = init_state
                .device()
                .create_fence(&vk::FenceCreateInfo::default(), None)?;

            let (blas, blas_buffer) = Self::create_blas(
                &acceleration_structure_loader,
                fence,
                init_state,
                pipeline_state,
                buffer_state,
            )?;
            let (tlas, tlas_buffer) = Self::create_tlas(
                &acceleration_structure_loader,
                fence,
                init_state,
                pipeline_state,
                blas,
            )?;

            let descriptor_pool = Self::create_descriptor_pool(init_state.device())?;
            let descriptor_sets = Self::create_descriptor_sets(
                init_state.device(),
                descriptor_pool,
                pipeline_state.descriptor_set_layout(),
            )?;

            let mut state = Self {
                loader: acceleration_structure_loader,
                fence,
                blas,
                blas_buffer,
                tlas,
                tlas_buffer,
                descriptor_pool,
                descriptor_sets,
            };
            state.update_descriptor_sets(
                init_state.device(),
                buffer_state.uniform_buffers(),
                swapchain_state.output_image_views(),
            );

            Ok(state)
        }
    }

    // unsafe fn create_acceleration_structure(
    //     acceleration_structure_loader: &acceleration_structure::Device,
    //     init_state: &InitState,
    //     pipeline_state: &PipelineState,
    //     buffer_state: &BufferState,
    // ) -> VkResult<(vk::AccelerationStructureKHR, Buffer<'a>)> {
    //     unimplemented!()
    // }

    unsafe fn create_blas(
        loader: &acceleration_structure::Device,
        fence: vk::Fence,
        init_state: &InitState,
        pipeline_state: &PipelineState,
        buffer_state: &BufferState,
    ) -> Result<(vk::AccelerationStructureKHR, Buffer<'a>), Box<dyn Error>> {
        let vertex_address = pipeline_state
            .buffer_device_address_loader()
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default()
                    .buffer(buffer_state.vertex_buffer().handle()),
            );

        let index_address = pipeline_state
            .buffer_device_address_loader()
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default()
                    .buffer(buffer_state.index_buffer().handle()),
            );

        println!("BLAS:");
        println!("\tVertex buffer address: {}", vertex_address);
        println!("\tIndex buffer address: {}", index_address);
        println!(
            "Vertex count: {}, Index count: {}",
            VERTICES.len(),
            INDICES.len()
        );

        let geometries = &[vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: vertex_address,
                    })
                    .vertex_stride(mem::size_of::<Vertex>() as vk::DeviceSize)
                    .max_vertex(VERTICES.len() as u32 - 1)
                    .index_type(vk::IndexType::UINT16)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: index_address,
                    }),
            })];

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(geometries);

        let primitive_count = INDICES.len() as u32 / 3;

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut size_info,
        );

        if size_info.acceleration_structure_size == 0 || size_info.build_scratch_size == 0 {
            println!(
                "BLAS size_info: accel_size={}, scratch_size={}",
                size_info.acceleration_structure_size, size_info.build_scratch_size
            );
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "BLAS build sizes are 0",
            )));
        }

        let buffer = Buffer::create(
            init_state.instance(),
            init_state.device(),
            init_state.physical_device(),
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let acceleration_structure = loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buffer.handle())
                .size(size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?;

        let mut scratch_buffer = Buffer::create(
            init_state.instance(),
            init_state.device(),
            init_state.physical_device(),
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let scratch_address = pipeline_state
            .buffer_device_address_loader()
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer.handle()),
            );

        let command_buffer = init_state.device().allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(init_state.queues().transfer().command_pool().unwrap())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];

        init_state.device().begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        build_info = build_info
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

        loader.cmd_build_acceleration_structures(
            command_buffer,
            &[build_info],
            &[&[vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(INDICES.len() as u32 / 3)]],
        );

        init_state.device().end_command_buffer(command_buffer)?;

        init_state.device().reset_fences(&[fence])?;
        init_state.device().queue_submit(
            init_state.queues().transfer().primary_handle().unwrap(),
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;

        init_state
            .device()
            .wait_for_fences(&[fence], true, u64::MAX)?;

        scratch_buffer.cleanup(init_state.device());

        init_state.device().free_command_buffers(
            init_state.queues().transfer().command_pool().unwrap(),
            &[command_buffer],
        );

        println!(
            "create_blas: BLAS handle={:?}, buffer={:?}",
            acceleration_structure,
            buffer.handle()
        );
        Ok((acceleration_structure, buffer))
    }

    unsafe fn create_tlas(
        loader: &acceleration_structure::Device,
        fence: vk::Fence,
        init_state: &InitState,
        pipeline_state: &PipelineState,
        blas: vk::AccelerationStructureKHR,
    ) -> Result<(vk::AccelerationStructureKHR, Buffer<'a>), Box<dyn Error>> {
        println!("create_tlas: 1");
        let instance = vk::AccelerationStructureInstanceKHR {
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: loader.get_acceleration_structure_device_address(
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

        println!(
            "create_tlas: Instance ref={:?}, transform={:?}, custom_index_mask={:?}, sbt_flags={:?}",
            instance.acceleration_structure_reference.device_handle,
            instance.transform.matrix,
            instance.instance_custom_index_and_mask,
            instance.instance_shader_binding_table_record_offset_and_flags
        );

        let bytes = slice::from_raw_parts(
            (&instance as *const _) as *const u8,
            mem::size_of_val(&instance),
        );
        println!("create_tlas: Bytes size={}, data={:?}", bytes.len(), bytes);

        let mut instances_buffer = Buffer::create_from_bytes_with_staging(
            init_state.instance(),
            init_state.device(),
            init_state.physical_device(),
            init_state.queues().command_fence().unwrap(),
            init_state.queues().transfer(),
            bytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        println!("create_tlas: Instances buffer created");

        let geometries = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: pipeline_state
                            .buffer_device_address_loader()
                            .get_buffer_device_address(
                                &vk::BufferDeviceAddressInfo::default()
                                    .buffer(instances_buffer.handle()),
                            ),
                    },
                ),
            })];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[1], // One instance (the cube BLAS)
            &mut size_info,
        );

        if size_info.acceleration_structure_size == 0 || size_info.build_scratch_size == 0 {
            println!(
                "TLAS size_info: accel_size={}, scratch_size={}",
                size_info.acceleration_structure_size, size_info.build_scratch_size
            );
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "TLAS build sizes are 0",
            )));
        }

        let tlas_buffer = Buffer::create(
            init_state.instance(),
            init_state.device(),
            init_state.physical_device(),
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let tlas = loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(tlas_buffer.handle())
                .size(size_info.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
            None,
        )?;

        let mut scratch_buffer = Buffer::create(
            init_state.instance(),
            init_state.device(),
            init_state.physical_device(),
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let scratch_address = pipeline_state
            .buffer_device_address_loader()
            .get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer.handle()),
            );

        let command_buffer = init_state.device().allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(init_state.queues().transfer().command_pool().unwrap())
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];

        init_state.device().begin_command_buffer(
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

        loader.cmd_build_acceleration_structures(
            command_buffer,
            &[build_info],
            &[&[vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)]],
        );

        init_state.device().end_command_buffer(command_buffer)?;

        init_state.device().reset_fences(&[fence])?;
        init_state.device().queue_submit(
            init_state.queues().transfer().primary_handle().unwrap(),
            &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
            fence,
        )?;

        init_state
            .device()
            .wait_for_fences(&[fence], true, u64::MAX)?;

        scratch_buffer.cleanup(init_state.device());
        instances_buffer.cleanup(init_state.device());

        init_state.device().free_command_buffers(
            init_state.queues().transfer().command_pool().unwrap(),
            &[command_buffer],
        );

        Ok((tlas, tlas_buffer))
    }

    unsafe fn create_descriptor_pool(device: &ash::Device) -> VkResult<vk::DescriptorPool> {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .pool_sizes(&[
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
                        .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
                        .ty(vk::DescriptorType::STORAGE_IMAGE),
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32)
                        .ty(vk::DescriptorType::UNIFORM_BUFFER),
                ])
                .max_sets(MAX_FRAMES_IN_FLIGHT as u32),
            None,
        )
    }

    unsafe fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> VkResult<Vec<vk::DescriptorSet>> {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&[descriptor_set_layout; MAX_FRAMES_IN_FLIGHT as usize]),
        )
    }

    pub fn update_descriptor_sets(
        &mut self,
        device: &ash::Device,
        uniform_buffers: &[Buffer],
        output_image_views: &[vk::ImageView],
    ) {
        unsafe {
            for (frame, &descriptor_set) in self.descriptor_sets.iter().enumerate() {
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
                                    .acceleration_structures(&[self.tlas]),
                            ),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .image_info(&[vk::DescriptorImageInfo::default()
                                .image_view(output_image_views[frame])
                                .image_layout(vk::ImageLayout::GENERAL)]),
                        vk::WriteDescriptorSet::default()
                            .dst_set(descriptor_set)
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .buffer_info(&[vk::DescriptorBufferInfo::default()
                                .buffer(uniform_buffers[frame].handle())
                                .offset(0)
                                .range(mem::size_of::<CameraGpu>() as u64)]),
                    ],
                    &[],
                );
            }
        }
    }

    pub fn cleanup(&mut self, init_state: &InitState) {
        unsafe {
            self.blas_buffer.cleanup(init_state.device());
            self.tlas_buffer.cleanup(init_state.device());
            init_state.device().destroy_fence(self.fence, None);

            self.loader.destroy_acceleration_structure(self.blas, None);
            self.loader.destroy_acceleration_structure(self.tlas, None);

            init_state
                .device()
                .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                .unwrap();
            init_state
                .device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
