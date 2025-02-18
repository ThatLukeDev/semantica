//! A semantic search library written in Rust.

#![warn(missing_docs)]

use rust_bert::pipelines::sentence_embeddings;

pub mod vector;

use crate::vector::Dot;

/// A 'hashmap' type object that semantically stores T by index of strings.
pub struct SemanticVec<T> {
    contents: Vec<(Vec<f32>, T)>
}

impl<T> SemanticVec<T> {
    pub fn search(&self, s: &str) -> T {
        let model = sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model()
            .unwrap();

        let embeddings = model.encode(&[s])
            .unwrap();

        todo!();
    }
}