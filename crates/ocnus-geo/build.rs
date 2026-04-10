#![allow(missing_docs)]
fn main() {
    cc::Build::new()
        .file("src/msis/nrlmsise-00.c")
        .file("src/msis/nrlmsise-00_data.c")
        .compile("nrlmsise-00");
}
