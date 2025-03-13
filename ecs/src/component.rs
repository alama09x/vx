pub trait Component: Send + Sync {}

pub trait Resource: Send + Sync + 'static {}
