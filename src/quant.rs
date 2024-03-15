use crate::{CPUTensor, DType, STORAGE_BUFFER_ALIGN};
use num::integer::div_floor;
use std::fmt::Debug;

/// Quantizer
///
/// Packs weights into our custom quantization formats.
#[derive(Debug, derive_new::new)]
pub struct Quantizer {
    format: Quantization,
}

impl Quantizer {
    pub fn quantize(&self, tensor: CPUTensor) -> CPUTensor {
        match self.format {
            Quantization::None => tensor,
            Quantization::SInt8 => self.sint8_quantize(tensor),
            Quantization::SInt4 => todo!(),
        }
    }

    pub fn dequantize(&self, tensor: CPUTensor) -> CPUTensor {
        match self.format {
            Quantization::None => tensor,
            Quantization::SInt8 => self.sint8_dequantize(tensor),
            Quantization::SInt4 => todo!(),
        }
    }

    /// Quantizes a float 32 tensor into a packed uint32 tensor.
    /// This is the rust equivalent of: https://www.w3.org/TR/WGSL/#pack4x8snorm-builtin
    /// This allows us to call `unpack4x8snorm` in the shader.
    /// It's a pretty naive quantization scheme, more to come.
    pub fn sint8_quantize(&self, tensor: CPUTensor) -> CPUTensor {
        let numel = tensor.shape().numel();
        assert!(numel % 4 == 0 && numel % 16 == 0);
        assert!(tensor.dt() == DType::F32); //TODO: f16, bf16
                                            //TODO: check if tensor is contiguous
        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();

        let qmatrix_len = numel / pack_size;
        let amatrix_len = numel / group_size;

        //returns the aligned number of ELEMENTS
        let aligner = |numel: usize, size_t: usize| -> usize {
            let nbytes = numel * size_t;
            let aligned = if nbytes % STORAGE_BUFFER_ALIGN != 0 {
                nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
            } else {
                nbytes
            };
            aligned / size_t
        };

        let mut quantized_matrix = vec![0u32; aligner(qmatrix_len, std::mem::size_of::<u32>())];
        let mut absmax_matrix = vec![0f32; aligner(amatrix_len, std::mem::size_of::<f32>())];

        let sf = 127.0f32;
        let mut block_absmax = f32::NEG_INFINITY;

        let matrix = tensor.to_vec::<f32>().unwrap();

        for i in (0..numel).step_by(pack_size) {
            if i % group_size == 0 {
                block_absmax = matrix[i..i + group_size]
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x.abs()));
            }
            let packed_value: i32 = ((matrix[i] / block_absmax * sf).round() as i32 & 0xFF)
                | (((matrix[i + 1] / block_absmax * sf).round() as i32 & 0xFF) << 8)
                | (((matrix[i + 2] / block_absmax * sf).round() as i32 & 0xFF) << 16)
                | (((matrix[i + 3] / block_absmax * sf).round() as i32 & 0xFF) << 24);
            quantized_matrix[i / pack_size] = packed_value as u32;
            absmax_matrix[i / group_size] = block_absmax;
        }
        quantized_matrix.append(&mut unsafe { std::mem::transmute(absmax_matrix) });
        unsafe { CPUTensor::from_quantized(quantized_matrix, tensor.shape().clone(), DType::WQ8) }
    }

    //TODO: this doesn't work
    pub fn sint8_dequantize(&self, quantized: CPUTensor) -> CPUTensor {
        assert!(quantized.dt() == DType::WQ8);
        let numel = quantized.shape().numel();
        let packed_numel = numel / self.format.pack_size() + numel / self.format.group_size();
        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();
        //Line below is invalid
        let quantized_matrix = quantized.to_vec::<u32>().unwrap();
        let mut dequantized = vec![0.0f32; numel];

        let absmax_start = packed_numel / group_size;

        for i in (0..packed_numel).step_by(pack_size) {
            let block_absmax = quantized_matrix[absmax_start + div_floor(i, group_size)] as f32;
            let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
            dequantized[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * block_absmax;
            dequantized[i + 3] = (packed_value >> 24) as f32 / 127.0 * block_absmax;
        }

        CPUTensor::from_slice(&dequantized, quantized.shape().clone())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    SInt8,
    SInt4,
}

impl Quantization {
    pub fn pack_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 4,
            Quantization::SInt4 => 8,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 16,
            Quantization::SInt4 => 8,
        }
    }
}
