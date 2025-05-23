// Inspired by Bevy's Transform implementation (MIT/Apache-2.0)

use std::slice;

use bevy_ecs::component::Component;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

use crate::IntoBytes;

#[derive(Component, Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TransformGpu {
    model: [[f32; 4]; 4],
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Default::default()
        }
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    #[inline]
    pub fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    #[inline]
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    #[inline]
    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }
}

impl TransformGpu {
    pub fn new(transform: &Transform) -> Self {
        Self {
            model: transform.to_mat4().to_cols_array_2d(),
        }
    }
}

impl IntoBytes for TransformGpu {
    fn to_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(slice::from_ref(self))
    }
}
