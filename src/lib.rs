//! A semantic search library written in Rust.

#![warn(missing_docs)]

use rust_bert::pipelines::sentence_embeddings::{self, SentenceEmbeddingsModel};

/// An implementation of vector operations on Vec.
pub mod vector;

use crate::vector::Dot;

const DIMENSION: usize = 384;
const TOLERANCE: f32 = 0.1;
const EMBEDDING_MINIMUM: f32 = 0.5;

/// An object that semantically stores T by index of strings.
pub struct SemanticVec<T> {
    contents: Vec<(Vec<f32>, T)>,

    ldts: Vec<(usize, f32)>,

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
            ldts: vec![],
            model: sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap(),
            ldt: (0..DIMENSION).map(|v| v as f32).collect()
        }
    }

    fn quick_search(&self, ldt: f32) -> usize {
        let len = self.ldts.len();
        if len <= 1 {
            return 0;
        }

        let mut start = 0;
        let mut end = len;
        let mut mid = (start + end) / 2;

        while start < end {
            if ldt > self.ldts[mid].1 {
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
        let embeddings = self.model.encode(&[id])
            .unwrap().concat();

        let ldt = embeddings.dot(&self.ldt).unwrap();

        let predicted = self.quick_search(ldt);

        self.contents.push((embeddings, val));

        self.ldts.insert(predicted, (self.contents.len() - 1, ldt));
    }

    /// Finds the closest match to input string.
    pub fn search(&self, s: &str) -> Option<&T> {
        let embeddings = self.model.encode(&[s])
            .ok()?.concat();

        let len = self.contents.len();

        let ldt = embeddings.dot(&self.ldt).ok()?;

        let mut predicted = self.quick_search(ldt);
        if predicted >= len {
            predicted = len - 1;
        }

        let predicted_dot = embeddings.dot(&self.contents[predicted].0).ok()?;

        let mut last_dot = predicted_dot;
        let mut best_dot = predicted_dot;
        let mut best_i = predicted;
        let mut i: i64 = 0;
        let mut actionable = true;

        while last_dot + TOLERANCE > best_dot && actionable {
            actionable = false;

            let low = predicted as i64 - i;
            let high = predicted as i64 + i;

            let mut best_in_part: usize = low as usize;
            let mut best_in_part_dot: f32 = 0.0;

            if low >= 0 {
                best_in_part_dot = embeddings.dot(&self.contents[low as usize].0).ok()?;

                actionable = true;
            }
            if high < len as i64 {
                let part_dot = embeddings.dot(&self.contents[high as usize].0).ok()?;

                if part_dot > best_in_part_dot {
                    best_in_part = high as usize;
                    best_in_part_dot = part_dot;
                }

                actionable = true;
            }

            last_dot = best_in_part_dot;

            if best_in_part_dot > best_dot {
                best_i = best_in_part;
                best_dot = best_in_part_dot;
            }

            i += 1;
        }

        match best_dot {
            EMBEDDING_MINIMUM.. => Some(&self.contents[best_i].1),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_search_internal() {
        let db = SemanticVec::<i32> {
            contents: vec![],
            ldts: vec![(0, 1.2), (0, 2.0), (0, 3.6), (0, 12.2), (0, 19.3)],
            model: sentence_embeddings::SentenceEmbeddingsBuilder::remote(sentence_embeddings::SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap(),
            ldt: (0..DIMENSION).map(|v| v as f32).collect()
        };

        assert_eq!(db.quick_search(12.2), 3);
        assert_eq!(db.quick_search(3.6), 2);
    }

    // TODO: impl tests
    #[test]
    fn semantic_search() {
        let mut db: SemanticVec<i32> = SemanticVec::new();

        db.add("One", 1);
        db.add("Two", 2);
        db.add("Three", 3);
        db.add("Four", 4);
        db.add("Five", 5);

        db.add("Dog", 808);
        db.add("Cat", 247);

        assert_eq!(
            db.search("1").unwrap().clone(),
            1
        );
        assert_eq!(
            db.search("2").unwrap().clone(),
            2
        );
        assert_eq!(
            db.search("3").unwrap().clone(),
            3
        );
        assert_eq!(
            db.search("4").unwrap().clone(),
            4
        );
        assert_eq!(
            db.search("5").unwrap().clone(),
            5
        );

        assert_eq!(
            db.search("feline").unwrap().clone(),
            247
        );

        assert_eq!(
            db.search("dinosaur"),
            None
        );
    }
}
