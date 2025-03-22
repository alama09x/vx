use std::f32;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec3};

use crate::{transform::Transform, Direction, IntoBytes};

#[derive(Clone, Copy)]
pub struct Camera {
    transform: Transform,
    fov: f32,
    window_width: f32,
    window_height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    pub view_projection: [[f32; 4]; 4],
}

impl Camera {
    const MOVE_SPEED: f32 = 3.0;

    const YAW_SPEED: f32 = 0.004;
    const PITCH_SPEED: f32 = 0.004;

    const PITCH_LIMIT: f32 = f32::consts::FRAC_PI_2 - 0.01;

    const SCROLL_SPEED: f32 = 3.0;
    const FOV_LIMIT_MIN: f32 = 1.0;
    const FOV_LIMIT_MAX: f32 = 179.0;

    pub fn new(transform: Transform, window_width: f32, window_height: f32) -> Self {
        Self {
            transform,
            fov: 45.0,
            window_width,
            window_height,
        }
    }

    pub fn zoom(&mut self, scroll: f32) {
        let degrees = scroll * 0.1 * Self::SCROLL_SPEED;
        self.fov = (self.fov - degrees).clamp(Self::FOV_LIMIT_MIN, Self::FOV_LIMIT_MAX);
    }

    pub fn move_in_direction(&mut self, direction: Direction, delta_time: f32) {
        let speed = Self::MOVE_SPEED * delta_time;

        let remove_y = Vec3::X + Vec3::Z;
        let local_x = (self.transform.rotation * Vec3::X * remove_y).normalize() * speed;
        let local_z = (self.transform.rotation * Vec3::Z * remove_y).normalize() * speed;

        match direction {
            Direction::Forward => self.transform.translation -= local_z,
            Direction::Back => self.transform.translation += local_z,
            Direction::Left => self.transform.translation -= local_x,
            Direction::Right => self.transform.translation += local_x,
            Direction::Up => self.transform.translation.y -= speed,
            Direction::Down => self.transform.translation.y += speed,
        }
    }

    pub fn rotate_by_mouse_movement(&mut self, dx: f32, dy: f32) {
        let dyaw = dx * Self::YAW_SPEED;
        let dpitch = -dy * Self::PITCH_SPEED;

        let (yaw, pitch, _roll) = self.transform.rotation.to_euler(EulerRot::YXZ);
        let yaw = yaw - dyaw;
        let pitch = (pitch - dpitch).clamp(-Self::PITCH_LIMIT, Self::PITCH_LIMIT);

        self.transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    }
}

impl IntoBytes for Camera {
    fn to_bytes(&self) -> Vec<u8> {
        let view = self.transform.to_mat4().inverse();

        let projection = Mat4::perspective_rh(
            self.fov.to_radians(),
            self.window_width / self.window_height,
            0.1,
            100.0,
        );

        let gpu_data = CameraGpu {
            view_projection: (projection * view).to_cols_array_2d(),
        };

        bytemuck::cast_slice(&[gpu_data]).to_vec()
    }
}
