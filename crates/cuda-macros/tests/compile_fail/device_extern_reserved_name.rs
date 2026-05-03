use cuda_macros::device;

#[device]
unsafe extern "C" {
    fn cuda_oxide_device_extern_evil();
}

fn main() {}
