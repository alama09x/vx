use std::f32;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec3};

use crate::{transform::Transform, IntoRaw, MoveDirection};

#[derive(Clone, Copy)]
pub struct Camera {
    transform: Transform,
    projection: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraRaw {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
}

impl Camera {
    pub fn new(transform: Transform, window_width: f32, window_height: f32) -> Self {
        let projection = Mat4::perspective_rh(
            45.0f32.to_radians(),
            window_width / window_height,
            0.1,
            100.0,
        );

        Self {
            transform,
            projection,
        }
    }

    pub fn translate(&mut self, translation: Vec3) {
        self.transform.translation += translation;
    }

    const MOVE_SPEED: f32 = 3.0;

    pub fn move_in_direction(&mut self, direction: MoveDirection, delta_time: f32) {
        let speed = Self::MOVE_SPEED * delta_time;
        match direction {
            MoveDirection::Forward => self.translate(Vec3::NEG_Z * speed),
            MoveDirection::Left => self.translate(Vec3::NEG_X * speed),
            MoveDirection::Back => self.translate(Vec3::Z * speed),
            MoveDirection::Right => self.translate(Vec3::X * speed),
            MoveDirection::Up => self.translate(Vec3::NEG_Y * speed),
            MoveDirection::Down => self.translate(Vec3::Y * speed),
        }
    }

    const YAW_SPEED: f32 = 0.004;
    const PITCH_SPEED: f32 = 0.004;
    const PITCH_LIMIT: f32 = f32::consts::FRAC_PI_2 - 0.01;

    pub fn rotate_by_mouse_movement(&mut self, dx: f32, dy: f32) {
        let dyaw = dx * Self::YAW_SPEED;
        let dpitch = -dy * Self::PITCH_SPEED;

        let (yaw, pitch, _roll) = self.transform.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw - dyaw;
        let pitch = (pitch - dpitch).clamp(-Self::PITCH_LIMIT, Self::PITCH_LIMIT);

        self.transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    }
}

impl IntoRaw for Camera {
    type Raw = CameraRaw;

    fn to_raw(&self) -> Self::Raw {
        Self::Raw {
            view: self.transform.to_mat4().inverse().to_cols_array_2d(),
            projection: self.projection.to_cols_array_2d(),
        }
    }
}
