use std::collections::HashSet;

use data::MoveDirection;
use gfx::state::VxState;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize},
    error::ExternalError,
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::CursorGrabMode,
};

pub struct VxApplication {
    app_name: &'static str,
    app_version: u32,
    title: String,
    size: LogicalSize<f32>,
    keys: HashSet<KeyCode>,
    state: Option<VxState>,
    suppress_next_mouse_delta: bool,
}

impl VxApplication {
    pub fn new(
        app_name: &'static str,
        app_version: u32,
        title: &str,
        width: f32,
        height: f32,
    ) -> Self {
        Self {
            app_name,
            app_version,
            title: title.into(),
            size: LogicalSize::new(width, height),
            keys: HashSet::new(),
            suppress_next_mouse_delta: false,
            state: None,
        }
    }

    fn capture_cursor(&self) -> Result<(), ExternalError> {
        if let Some(ref state) = self.state {
            state.window.set_cursor_position(LogicalPosition::new(
                self.size.width * 0.5,
                self.size.height * 0.5,
            ))?;

            state.window.set_cursor_visible(false);

            state
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))?;
        }

        Ok(())
    }

    fn release_cursor(&self) -> Result<(), ExternalError> {
        if let Some(ref state) = self.state {
            state.window.set_cursor_visible(true);
            state.window.set_cursor_grab(CursorGrabMode::None)?;
        }

        Ok(())
    }

    fn direction_from_key(key: KeyCode) -> Option<MoveDirection> {
        match key {
            KeyCode::KeyW => Some(MoveDirection::Forward),
            KeyCode::KeyA => Some(MoveDirection::Left),
            KeyCode::KeyS => Some(MoveDirection::Back),
            KeyCode::KeyD => Some(MoveDirection::Right),
            KeyCode::Space => Some(MoveDirection::Up),
            KeyCode::ShiftLeft => Some(MoveDirection::Down),
            _ => None,
        }
    }
}

impl ApplicationHandler for VxApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let state = VxState::new(
                self.app_name,
                self.app_version,
                &self.title,
                &self.size,
                event_loop,
            )
            .unwrap();
            self.state = Some(state);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.suppress_next_mouse_delta {
                self.suppress_next_mouse_delta = false;
                return;
            }

            if let Some(ref mut state) = self.state {
                let (dx, dy) = delta;
                state.camera.rotate_by_mouse_movement(dx as f32, dy as f32);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.state = None;
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                if let Some(ref mut state) = self.state {
                    for &key in &self.keys {
                        if let Some(direction) = Self::direction_from_key(key) {
                            state.camera.move_in_direction(direction, state.delta_time);
                        }
                    }
                    state.draw_frame().unwrap();
                    state.window.request_redraw();
                }
            }

            WindowEvent::Focused(focused) => {
                if focused {
                    self.capture_cursor().unwrap();
                } else {
                    self.release_cursor().unwrap();
                }
                self.suppress_next_mouse_delta = focused;
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape {
                        self.state = None;
                        event_loop.exit();
                    }
                    match event.state {
                        ElementState::Pressed => self.keys.insert(key),
                        ElementState::Released => self.keys.remove(&key),
                    };
                }
            }

            WindowEvent::Resized(_) => {
                if let Some(ref mut state) = self.state {
                    state.recreate_swapchain().unwrap();
                }
            }
            _ => (),
        }
    }
}
