use std::{fs::File, io::Read};

pub struct IdxFile {
    dim_size: usize,
    dim_sizes: Vec<u32>,
    data_type: u8,
    pub data_size: u8,
    pub data: Vec<u8>,
}
impl IdxFile {
    pub fn load(path: &str) -> IdxFile {
        let mut buf: Vec<u8> = vec![];

        let mut file = Box::new(File::open(path).unwrap());
        file.read_to_end(&mut buf).unwrap();

        let data_type: u8 = buf[2];
        let data_size: u8 = match data_type {
            0x08 => 1,
            0x09 => 1,
            0x0B => 2,
            0x0C => 4,
            0x0D => 4,
            0x0E => 8,
            _ => 0,
        };
        let dim_size: usize = buf[3] as usize;

        let mut dim_sizes: Vec<u32> = vec![];
        for i in 0..dim_size {
            let mut dim: u32 = 0;
            for j in 0..4 {
                dim |= (buf[4 + i * 4 + j] as u32) << (3 - j as u32) * 8;
            }
            dim_sizes.push(dim);
        }
        
        let data: Vec<u8> = buf[(4 + dim_size * 4)..].to_vec();

        IdxFile {
            dim_size,
            dim_sizes,
            data_type,
            data_size,
            data,
        }
    }

    pub fn print_img(&self, img_id: usize) {
        assert!(self.dim_size == 3 && self.dim_sizes[1] == 28 && self.dim_sizes[2] == 28 && self.dim_sizes[0] > img_id as u32 && self.data_size == 1);

        for i in 0..28 {
            for j in 0..28 {
                if self.data[i * 28 + j + 28 * 28 * img_id] == 0 {
                    print!(" ");
                } else {
                    print!("-");
                }
            }
            println!();
        }
    }

    pub fn print_label(&self, label_id: usize) {
        assert!(self.dim_size == 1 && self.dim_sizes[0] > label_id as u32);

        println!("{}", self.data[label_id]);
    }

    pub fn get_image(&self, img_id: usize) -> Vec<u8> {
        self.data[(img_id * 784)..((img_id + 1) * 784)].to_vec()
    }
    pub fn get_byte(&self, byte_pos: usize) -> u8 {
        self.data[byte_pos]
    }
}