use std::marker::PhantomData;

pub struct Query<C, F = ()> {
    phantom_data: PhantomData<(C, F)>,
}
