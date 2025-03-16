use app::VxApplication;
use winit::event_loop::EventLoop;

fn main() {
    let mut app = VxApplication::new("Hello World", 0, "Hello World", 800.0, 600.0);
    let event_loop = EventLoop::new().unwrap();

    event_loop.run_app(&mut app).unwrap();
}
