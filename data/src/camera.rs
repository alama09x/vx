use std::{f32, slice};

use bevy_ecs::component::Component;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::{transform::Transform, IntoBytes};

#[derive(Component, Clone, Copy)]
#[require(Transform, CameraFov)]
pub struct Camera;

#[derive(Component, Clone, Copy)]
pub struct CameraFov(f32);

impl Default for CameraFov {
    fn default() -> Self {
        Self::from_degrees(45.0)
    }
}

impl CameraFov {
    const LIMIT_MIN: f32 = 1.0;
    const LIMIT_MAX: f32 = 179.0;

    pub fn from_radians(radians: f32) -> Self {
        Self(radians.to_degrees())
    }

    pub fn from_degrees(degrees: f32) -> Self {
        Self(degrees)
    }

    pub fn radians(&self) -> f32 {
        self.0.to_radians()
    }

    pub fn degrees(&self) -> f32 {
        self.0
    }

    pub fn zoom(&mut self, scroll: f32, scroll_speed: f32) {
        let degrees = scroll * 0.1 * scroll_speed;
        self.0 = (self.0 - degrees).clamp(Self::LIMIT_MIN, Self::LIMIT_MAX);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    pub proj_inverse: [[f32; 4]; 4],
    pub view_inverse: [[f32; 4]; 4],
}

impl CameraGpu {
    pub fn new(
        transform: &Transform,
        fov_degrees: f32,
        window_width: f32,
        window_height: f32,
    ) -> Self {
        let view_inverse = transform.to_mat4().to_cols_array_2d();

        let proj_inverse = Mat4::perspective_rh(
            fov_degrees.to_radians(),
            window_width / window_height,
            0.1,
            100.0,
        )
        .inverse()
        .to_cols_array_2d();

        CameraGpu {
            view_inverse,
            proj_inverse,
        }
    }
}

impl IntoBytes for CameraGpu {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(slice::from_ref(self))
    }
}
