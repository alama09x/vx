use std::mem;

use bevy_ecs::system::Resource;
use bytemuck::{Pod, Zeroable};
use data::camera::CameraGpu;

mod buffer;

pub mod acceleration_structure_state;
pub mod buffer_state;
pub mod command_state;
pub mod init_state;
pub mod pipeline_state;
pub mod swapchain_state;

const MAX_FRAMES_IN_FLIGHT: u8 = 2;

const UNIFORM_BUFFER_SIZE: usize = mem::size_of::<CameraGpu>();

const VERTICES: [Vertex; 3] = [
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
];
//     Vertex {
//         pos: [-0.5, 0.5, 0.5],
//         color: [1.0, 0.0, 1.0],
//     },
//     // Back
//     Vertex {
//         pos: [-0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 0.0],
//     },
//     Vertex {
//         pos: [-0.5, -0.5, -0.5],
//         color: [0.0, 1.0, 0.0],
//     },
//     Vertex {
//         pos: [0.5, -0.5, -0.5],
//         color: [0.0, 0.0, 1.0],
//     },
//     Vertex {
//         pos: [0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 1.0],
//     },
//     // Bottom
//     Vertex {
//         pos: [0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 0.0],
//     },
//     Vertex {
//         pos: [0.5, 0.5, 0.5],
//         color: [0.0, 1.0, 0.0],
//     },
//     Vertex {
//         pos: [-0.5, 0.5, 0.5],
//         color: [0.0, 0.0, 1.0],
//     },
//     Vertex {
//         pos: [-0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 1.0],
//     },
//     // Top
//     Vertex {
//         pos: [0.5, -0.5, 0.5],
//         color: [1.0, 0.0, 0.0],
//     },
//     Vertex {
//         pos: [0.5, -0.5, -0.5],
//         color: [0.0, 1.0, 0.0],
//     },
//     Vertex {
//         pos: [-0.5, -0.5, -0.5],
//         color: [0.0, 0.0, 1.0],
//     },
//     Vertex {
//         pos: [-0.5, -0.5, 0.5],
//         color: [1.0, 0.0, 1.0],
//     },
//     // Right
//     Vertex {
//         pos: [0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 0.0],
//     },
//     Vertex {
//         pos: [0.5, -0.5, -0.5],
//         color: [0.0, 1.0, 0.0],
//     },
//     Vertex {
//         pos: [0.5, -0.5, 0.5],
//         color: [0.0, 0.0, 1.0],
//     },
//     Vertex {
//         pos: [0.5, 0.5, 0.5],
//         color: [1.0, 0.0, 1.0],
//     },
//     // Left
//     Vertex {
//         pos: [-0.5, 0.5, 0.5],
//         color: [1.0, 0.0, 0.0],
//     },
//     Vertex {
//         pos: [-0.5, -0.5, 0.5],
//         color: [0.0, 1.0, 0.0],
//     },
//     Vertex {
//         pos: [-0.5, -0.5, -0.5],
//         color: [0.0, 0.0, 1.0],
//     },
//     Vertex {
//         pos: [-0.5, 0.5, -0.5],
//         color: [1.0, 0.0, 1.0],
//     },
// ];

const INDICES: [u16; 3] = [0, 1, 2];

// const INDICES: [u16; 6 * 6] = [
//     0, 1, 2, 0, 2, 3, // Front
//     4, 5, 6, 4, 6, 7, // Back
//     8, 9, 10, 8, 10, 11, // Bottom
//     12, 13, 14, 12, 14, 15, // Top
//     16, 17, 18, 16, 18, 19, // Right
//     20, 21, 22, 20, 22, 23, // Left
// ];

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Resource, Default)]
pub struct CurrentFrame(pub u8);

impl CurrentFrame {
    pub fn next(&self) -> u8 {
        (self.0 + 1) % MAX_FRAMES_IN_FLIGHT
    }
}
