//! A semantic search library written in Rust.

#![warn(missing_docs)]

use rust_bert::pipelines::sentence_embeddings::{self, SentenceEmbeddingsModel};

/// An implementation of vector operations on Vec.
pub mod vector;

use crate::vector::Dot;

const DIMENSION: usize = 384;
const DIFFERENCE: f32 = 0.1;

/// A 'hashmap' type object that semantically stores T by index of strings.
pub struct SemanticVec<'a, T> {
    contents: Vec<(Vec<f32>, T)>,

    buckets: [Vec<&'a (Vec<f32>, T)>; DIMENSION],

    model: SentenceEmbeddingsModel,
}

impl<'a, T> SemanticVec<'a, T> {
    /// Creates a new SemanticVec
    pub fn new() -> Self
    where T: std::fmt::Debug {
        SemanticVec::<'a, _> {
            contents: vec![],
            buckets: vec![vec![]; DIMENSION].try_into().unwrap(),
            model: sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap()
        }
    }

    /// Adds value to SemanticVec, and recalculates buckets.
    pub fn add(&'a mut self, id: &str, val: T) {
        let embeddings = self.model.encode(&[id])
            .unwrap()[0].clone();

        let mut buckets = vec!{};
        let mut largest = 0.0;

        for i in 0..DIMENSION {
            let difference = embeddings[i] - largest;

            if difference > 0.0 {
                largest = embeddings[i];

               if difference > DIFFERENCE {
                    buckets.clear();
                }

                buckets.push(i);
            }
        }

        self.contents.push((embeddings, val));

        for i in buckets {
            self.buckets[i].push(self.contents.last().unwrap());
        }
    }

    /// Finds the closest match to input string.
    pub fn search(&self, s: &str) -> Option<&T> {
        let embeddings = self.model.encode(&[s])
            .unwrap();

        let mut buckets = vec!{};
        let mut largest = 0.0;

        for i in 0..DIMENSION {
            let difference = embeddings[0][i] - largest;

            if difference > 0.0 {
                largest = embeddings[0][i];

                if difference > DIFFERENCE {
                    buckets.clear();
                }

                buckets.push(i);
            }
        }

        let mut closest_match = None;
        let mut closest_cos_dot = 0.0;

        for i in buckets {
            for item in &self.buckets[i] {
                let cos_dot = item.0.dot(&embeddings[0]).unwrap();

                if cos_dot > closest_cos_dot {
                    closest_cos_dot = cos_dot;
                    closest_match = Some(&item.1);
                }
            }
        }

        closest_match
    }
}
