use gfx::state::VxState;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};

pub struct VxApplication {
    app_name: &'static str,
    app_version: u32,
    title: String,
    size: LogicalSize<f32>,
    state: Option<VxState>,
    window: Option<Window>,
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
            state: None,
            window: None,
        }
    }
}

impl ApplicationHandler for VxApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title(&self.title)
                    .with_inner_size(self.size),
            )
            .expect("Failed to create window");
        self.state = Some(VxState::new(self.app_name, self.app_version, &window).unwrap());
        println!("Initialized");
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        }
    }
}
