use half::f16;
use num_traits::{AsPrimitive, Float};

use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    SInt8,
    SInt4,
    Float16,
}

impl Quantization {
    pub fn allowed_mae(&self) -> f32 {
        match self {
            Quantization::None => 1e-5,
            Quantization::Float16 => 1e-3,
            Quantization::SInt8 => 1e-2,
            Quantization::SInt4 => 1e1,
        }
    }
}

pub fn sint4_quantize<F: Float + AsPrimitive<i32> + Debug>(
    matrix: &[F],
    K: usize,
    N: usize,
) -> (Vec<u32>, F) {
    assert!(matrix.len() == K * N);
    assert!(matrix.len() % 4 == 0);
    let pack_size = 8;
    let mut quantized_matrix = vec![0u32; K * N / pack_size];

    let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
    let sf = F::from(7.).unwrap();

    for i in (0..(K * N)).step_by(pack_size) {
        let packed_value: i32 = ((matrix[i] / absmax * sf).round().as_() & 0xF)
            | (((matrix[i + 1] / absmax * sf).round().as_() & 0xF) << 4)
            | (((matrix[i + 2] / absmax * sf).round().as_() & 0xF) << 8)
            | (((matrix[i + 3] / absmax * sf).round().as_() & 0xF) << 12)
            | (((matrix[i + 4] / absmax * sf).round().as_() & 0xF) << 16)
            | (((matrix[i + 5] / absmax * sf).round().as_() & 0xF) << 20)
            | (((matrix[i + 6] / absmax * sf).round().as_() & 0xF) << 24)
            | (((matrix[i + 7] / absmax * sf).round().as_() & 0xF) << 28);
        quantized_matrix[i / pack_size] = packed_value as u32
    }
    (quantized_matrix, absmax)
}

pub fn sint4_dequantize(quantized_matrix: &[u32], absmax: f32, K: usize, N: usize) -> Vec<f32> {
    let pack_size = 8;
    let mut matrix = vec![0.0; K * N];

    for i in (0..(K * N)).step_by(pack_size) {
        let packed_value = quantized_matrix[i.div_floor(pack_size)] as i32;
        matrix[i] = ((packed_value << 28) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 1] = ((packed_value << 24) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 2] = ((packed_value << 20) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 3] = ((packed_value << 16) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 4] = ((packed_value << 12) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 5] = ((packed_value << 8) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 6] = ((packed_value << 4) >> 28) as f32 / 7.0 * absmax;
        matrix[i + 7] = (packed_value >> 28) as f32 / 7.0 * absmax;
    }

    matrix
}

/// Quantize a matrix of floats to 8-bit signed integers.
/// The AsPrimitive<i32> may seem confusing, we be need to do the bit masking
/// using signed integers, then cast to unsigned, to avoid losing negative values
/// TODO: enhance with group level absmax
pub fn sint8_quantize<F: Float + AsPrimitive<i32> + Debug>(
    matrix: &[F],
    K: usize,
    N: usize,
) -> (Vec<u32>, F) {
    assert!(matrix.len() == K * N);
    assert!(matrix.len() % 4 == 0);
    let pack_size = 4;
    let mut quantized_matrix = vec![0u32; K * N / pack_size];

    let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
    let sf = F::from(127.).unwrap();

    for i in (0..(K * N)).step_by(pack_size) {
        let packed_value: i32 = ((matrix[i] / absmax * sf).round().as_() & 0xFF)
            | (((matrix[i + 1] / absmax * sf).round().as_() & 0xFF) << 8)
            | (((matrix[i + 2] / absmax * sf).round().as_() & 0xFF) << 16)
            | (((matrix[i + 3] / absmax * sf).round().as_() & 0xFF) << 24);
        quantized_matrix[i / pack_size] = packed_value as u32
    }
    (quantized_matrix, absmax)
}

pub fn sint8_dequantize(quantized_matrix: &[u32], absmax: f32, K: usize, N: usize) -> Vec<f32> {
    let pack_size = 4;
    let mut matrix = vec![0.0; K * N];

    for i in (0..(K * N)).step_by(pack_size) {
        let packed_value = quantized_matrix[i.div_floor(pack_size)] as i32;
        matrix[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 3] = (packed_value >> 24) as f32 / 127.0 * absmax;
    }

    matrix
}

/// Quantize a matrix of 32 bit floats to a packed representation of 16 bit floats
/// We pack 2 floats into a single u32
pub fn float16_quantize(matrix: &[f32], K: usize, N: usize) -> Vec<u32> {
    assert!(matrix.len() == K * N);
    assert!(matrix.len() % 2 == 0);

    let mut result = Vec::with_capacity(matrix.len() / 2);

    for floats in matrix.chunks(2) {
        let float0 = f16::from_f32(floats[0]).to_bits() as u32;
        let float1 = f16::from_f32(floats[1]).to_bits() as u32;
        let packed = float1 << 16 | float0;
        result.push(packed);
    }

    result
}

//Dequantize a matrix of 16 bit floats to 32 bit floats
pub fn float16_dequantize(matrix: &[u32], _K: usize, _N: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(matrix.len() * 2);

    for packed in matrix {
        let float2 = f16::from_bits((packed >> 16) as u16).to_f32();
        let float1 = f16::from_bits((packed & 0xFFFF) as u16).to_f32();
        result.push(float1);
        result.push(float2);
    }

    result
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn test_sint8_qdq() {
        let matrix = vec![
            0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2,
        ];
        let (quantized_matrix, absmax) = super::sint8_quantize(&matrix, 4, 4);
        assert_eq!(quantized_matrix.len(), 4);
        assert_eq!(
            quantized_matrix,
            vec![3409310987, 2172622442, 3409310987, 2172622442]
        );
        let dequantized_matrix = super::sint8_dequantize(&quantized_matrix, absmax, 4, 4);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.01);
        }
    }

    #[test]
    pub fn test_sint4_qdq() {
        let matrix = vec![
            0.1, -0.1, 0.6, -0.5, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2,
        ];
        println!("{:?}", matrix);
        let (quantized_matrix, absmax) = super::sint4_quantize(&matrix, 4, 4);
        assert_eq!(quantized_matrix.len(), 2);
        assert_eq!(quantized_matrix, vec![2544293105, 2544292849]);
        let dequantized_matrix = super::sint4_dequantize(&quantized_matrix, absmax, 4, 4);
        println!("{:?}", dequantized_matrix);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.1);
        }
    }

    #[test]
    pub fn test_float16_qdq() {
        let matrix = vec![
            0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2,
        ];
        println!("{:?}", matrix);
        let quantized_matrix = super::float16_quantize(&matrix, 4, 4);
        println!("{:?}", quantized_matrix);
        assert_eq!(quantized_matrix.len(), 8);
        assert_eq!(
            quantized_matrix,
            vec![
                2925932134, 3087022080, 3154131968, 3167567053, 2925932134, 3087022080, 3154131968,
                3167567053
            ]
        );
        let dequantized_matrix = super::float16_dequantize(&quantized_matrix, 4, 4);
        println!("{:?}", dequantized_matrix);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.001);
        }
    }
}
