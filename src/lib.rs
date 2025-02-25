//! A semantic search library written in Rust.

#![warn(missing_docs)]

use rust_bert::pipelines::sentence_embeddings::{self, SentenceEmbeddingsModel};

/// An implementation of vector operations on Vec.
pub mod vector;

use crate::vector::Dot;

const DIMENSION: usize = 384;

/// An object that semantically stores T by index of strings.
pub struct SemanticVec<T> {
    contents: Vec<(Vec<f32>, T, f32)>,

    model: SentenceEmbeddingsModel,

    /// A linear dot transformer.
    /// Typically should be in the form
    /// `vec!![0..DIMENSION]`
    ldt: Vec<f32>,
}

impl<T: Clone> SemanticVec<T> {
    /// Creates a new SemanticVec
    pub fn new() -> Self
    where T: std::fmt::Debug {
        SemanticVec::<_> {
            contents: vec![],
            model: sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap(),
            ldt: (0..DIMENSION).map(|v| v as f32).collect()
        }
    }

    fn quick_search(&self, ldt: f32) -> usize {
        let mut start = 0;
        let mut end = self.contents.len();
        let mut mid = (start + end) / 2;

        while start < end {
            if ldt > self.contents[mid].2 {
                start = mid + 1;
            }
            else {
                end = mid;
            }

            mid = (start + end) / 2;
        }

        mid
    }

    /// Adds value to SemanticVec, and recalculates linear dot product.
    pub fn add(&mut self, id: &str, val: T) {
        todo!();

        let embeddings = self.model.encode(&[id])
            .unwrap().concat();

        let ldt = embeddings.dot(&self.ldt).unwrap();

        let entry = (embeddings, val, ldt);

        let predicted = self.quick_search(ldt);
    }

    /// Finds the closest match to input string.
    pub fn search(&self, s: &str) -> Option<&T> {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_search_internal() {
        let db = SemanticVec::<i32> {
            contents: vec![(vec![], 0, 1.2), (vec![], 0, 2.0), (vec![], 0, 3.6), (vec![], 0, 12.2), (vec![], 0, 19.3)],
            model: sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap(),
            ldt: (0..DIMENSION).map(|v| v as f32).collect()
        };

        assert_eq!(db.quick_search(12.2), 3);
        assert_eq!(db.quick_search(3.6), 2);
    }

    // TODO: impl tests
}
