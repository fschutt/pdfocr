//! Text post-processing: the long s (ſ) and words split by a line-end hyphen.
//!
//! 18th-century print sets a long s everywhere except at the end of a word ("ſhould", "Moſes").
//! Tesseract's `enm` model reads it as `ſ`; most other engines read it as `f` ("fhould", "Mofes").
//! [`LongS::Repair`] turns `ſ` into `s` and fixes `f` read for a long s with a word list: a word
//! that is not in the list becomes the first variant with some of its `f` turned into `s` that is.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum LongS {
    /// Leave the text as the engine read it
    Keep,
    /// Replace ſ with s
    S,
    /// Replace ſ with s, and f read for a long s with s where the word list says so
    Repair,
    /// As repair, for text a reader already corrected (`pdf-ocr-bench reconstruct`): only an f
    /// that is no word's (no fixed pairs: "fame" may be fame), and an f before a vowel only in
    /// a lower-case word of five letters or more (Latin "fide", "fuit", "fol.", names "Rufin.")
    Careful,
}

/// Word pairs where both spellings are words but the f-form is rare in 18th-century prose, so an
/// engine that reads a long s as f almost always means the s-form. Pairs that are both common
/// (faith/saith, fold/sold, fight/sight) are left alone.
/// Words with an f that a word list of base forms lacks (comparatives): never "repaired".
const KNOWN_F: &[&str] = &["fewer", "fewest", "finer", "finest", "fuller", "fullest", "fairer", "fairest", "fatter", "firmer"];

const PREFER_S: &[(&str, &str)] = &[
    ("fo", "so"),
    ("fame", "same"),
    ("fent", "sent"),
    ("fet", "set"),
    ("fide", "side"),
    ("fides", "sides"),
    ("fin", "sin"),
    ("fins", "sins"),
    ("fon", "son"),
    ("fons", "sons"),
    ("fun", "sun"),
    ("fum", "sum"),
    ("fee", "see"),
    ("fees", "sees"),
    ("fays", "says"),
    ("faying", "saying"),
];

/// A word list such as /usr/share/dict/words, with regular inflections.
pub struct Dictionary {
    lower: HashSet<String>,
    /// Lower-cased entries that the list capitalizes (names): they only match capitalized words.
    proper: HashSet<String>,
}

impl Dictionary {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading word list {}", path.display()))?;
        Ok(Self::from_words(text.lines()))
    }

    pub fn from_words<'a>(words: impl IntoIterator<Item = &'a str>) -> Self {
        let mut lower = HashSet::new();
        let mut proper = HashSet::new();
        for word in words.into_iter().map(str::trim).filter(|w| !w.is_empty()) {
            if word.chars().next().is_some_and(char::is_uppercase) {
                proper.insert(word.to_lowercase());
            } else {
                lower.insert(word.to_string());
            }
        }
        Self { lower, proper }
    }

    pub fn len(&self) -> usize {
        self.lower.len() + self.proper.len()
    }

    /// `word` or a regular inflection of a listed word: walked, receiving, addreſs'd, Moses's.
    /// Names only match when `word` is capitalized.
    pub fn knows(&self, word: &str) -> bool {
        let capitalized = word.chars().next().is_some_and(char::is_uppercase);
        let lower = word.to_lowercase();
        base_forms(&lower)
            .iter()
            .any(|base| self.lower.contains(base) || (capitalized && self.proper.contains(base)))
    }
}

/// `word` and the stems it may be an inflection of (all lower case).
fn base_forms(word: &str) -> Vec<String> {
    let mut out = vec![word.to_string()];
    for apostrophe in ["'s", "'d", "'st", "'n", "'t"] {
        if let Some(stem) = word.strip_suffix(apostrophe) {
            out.push(stem.to_string());
            out.push(format!("{stem}e")); // lov'd
        }
    }
    for (suffix, replacement) in [("ies", "y"), ("ied", "y")] {
        if let Some(stem) = word.strip_suffix(suffix) {
            out.push(format!("{stem}{replacement}"));
        }
    }
    // only the inflections that are regular for almost every word: a looser rule ("-est" on
    // "perse") turns OCR noise into a "word" and the repair into a new error
    for suffix in ["s", "es", "ed", "d", "ing", "ly"] {
        let Some(stem) = word.strip_suffix(suffix) else { continue };
        if stem.len() < 3 {
            continue;
        }
        out.push(stem.to_string());
        if matches!(suffix, "ed" | "ing") {
            out.push(format!("{stem}e")); // received, receiving
            let b = stem.as_bytes();
            if b.len() >= 3 && b[b.len() - 1] == b[b.len() - 2] {
                out.push(stem[..stem.len() - 1].to_string()); // stopped
            }
        }
    }
    // British -our spellings, which an American word list has as -or: favour, favourable. Only
    // at the end of the word (or before -able, -ite..): "fource" is "source", not "force".
    for base in out.clone() {
        for tail in ["", "able", "ably", "ite", "ites", "ful", "less"] {
            if let Some(stem) = base.strip_suffix(&format!("our{tail}")) {
                if stem.len() >= 2 {
                    out.push(format!("{stem}or{tail}"));
                }
            }
        }
    }
    out
}

pub fn replace_long_s(text: &str) -> String {
    text.replace('ſ', "s").replace('ﬅ', "st")
}

/// `word` (with its punctuation) with `f` read for a long s turned into `s`, or None if it
/// needs no change: "fhould," -> "should,", "Addreffes" -> "Addresses", "Mofes's" -> "Moses's".
pub fn repair_f(word: &str, dict: &Dictionary) -> Option<String> {
    repair_f_with(word, dict, false, false)
}

/// `repair_f`, or with `careful` its cautious form (see `LongS::Careful`); an `italic` word (Latin
/// in this kind of book: "feras", "fecit") only where an f stands before a consonant.
pub fn repair_f_with(word: &str, dict: &Dictionary, careful: bool, italic: bool) -> Option<String> {
    let start = word.find(|c: char| c.is_alphabetic())?;
    let end = word.char_indices().rev().find(|&(_, c)| c.is_alphabetic() || c == '\'').map(|(i, c)| i + c.len_utf8())?;
    let core = &word[start..end];
    if !core.contains('f') {
        return None;
    }
    let lower = core.to_lowercase();
    if let Some((_, s_form)) = PREFER_S.iter().find(|(f_form, _)| *f_form == lower).filter(|_| !careful) {
        return Some(format!("{}{}{}", &word[..start], match_case(core, s_form), &word[end..]));
    }
    if dict.knows(core) || KNOWN_F.contains(&lower.as_str()) {
        return None;
    }
    let chars: Vec<char> = core.chars().collect();
    // a long s never ends a word, so neither does an f read for one
    let last_letter = chars.iter().rposition(|c| c.is_alphabetic())?;
    let long_lower = !italic && chars.iter().filter(|c| c.is_alphabetic()).count() >= 5 && chars[0].is_lowercase();
    let positions: Vec<usize> = (0..last_letter)
        .filter(|&i| chars[i] == 'f')
        // ſt, ſh, ſp.. are long-s spellings in any language; ſa, ſe, ſi.. also begin Latin words
        .filter(|&i| !careful || long_lower || !"aeiouy".contains(chars[i + 1].to_ascii_lowercase()))
        .collect();
    for n in 1..=positions.len().min(3) {
        for combo in combinations(&positions, n) {
            let variant: String = chars.iter().enumerate().map(|(i, &c)| if combo.contains(&i) { 's' } else { c }).collect();
            if dict.knows(&variant) {
                return Some(format!("{}{}{}", &word[..start], variant, &word[end..]));
            }
        }
    }
    None
}

fn match_case(original: &str, replacement: &str) -> String {
    if original.chars().next().is_some_and(char::is_uppercase) {
        let mut chars = replacement.chars();
        chars.next().map(|c| c.to_uppercase().chain(chars).collect()).unwrap_or_default()
    } else {
        replacement.to_string()
    }
}

fn combinations(items: &[usize], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    (0..items.len())
        .flat_map(|i| {
            combinations(&items[i + 1..], n - 1).into_iter().map(move |mut rest| {
                rest.insert(0, items[i]);
                rest
            })
        })
        .collect()
}

/// Applies the `--long-s` mode to words and counts what it changed.
pub struct Fixer<'a> {
    pub mode: LongS,
    pub dict: Option<&'a Dictionary>,
    pub long_s: usize,
    pub repaired: usize,
    pub examples: HashMap<String, String>,
}

impl<'a> Fixer<'a> {
    pub fn new(mode: LongS, dict: Option<&'a Dictionary>) -> Self {
        Self { mode, dict, long_s: 0, repaired: 0, examples: HashMap::new() }
    }

    pub fn word(&mut self, word: &str) -> String {
        self.word_styled(word, false)
    }

    /// `word`, knowing whether it is set in italic (see `repair_f_with`).
    pub fn word_styled(&mut self, word: &str, italic: bool) -> String {
        if self.mode == LongS::Keep {
            return word.to_string();
        }
        let mut out = replace_long_s(word);
        if out != word {
            self.long_s += 1;
        }
        if let (LongS::Repair | LongS::Careful, Some(dict)) = (self.mode, self.dict) {
            if let Some(fixed) = repair_f_with(&out, dict, self.mode == LongS::Careful, italic) {
                self.repaired += 1;
                if self.examples.len() < 12 {
                    self.examples.insert(out.clone(), fixed.clone());
                }
                out = fixed;
            }
        }
        out
    }
}

/// Joins a word split by a line-end hyphen ("du-" + "ring,") or returns None when the second
/// part does not continue a word (it starts with an upper-case letter or a digit). The hyphen is
/// dropped unless both parts are words and the joined form is not ("Ear-" + "rings" stays
/// "Ear-rings" if "earrings" is not listed).
pub fn join_hyphenated(first: &str, second: &str, dict: Option<&Dictionary>) -> Option<String> {
    let stem = first.strip_suffix('-').or_else(|| first.strip_suffix('¬'))?;
    if !stem.chars().last().is_some_and(char::is_alphabetic) || !second.chars().next().is_some_and(char::is_lowercase) {
        return None;
    }
    let Some(dict) = dict else { return Some(format!("{stem}{second}")) };
    let letters = |w: &str| w.trim_matches(|c: char| !c.is_alphabetic() && c != '\'').to_string();
    let joined = letters(&format!("{stem}{second}"));
    let clean = |w: &str| replace_long_s(w);
    let known = |w: &str| dict.knows(&clean(w)) || repair_f(&clean(w), dict).is_some();
    if !dict.knows(&clean(&joined)) && known(&letters(stem)) && known(&letters(second)) && repair_f(&clean(&joined), dict).is_none() {
        return Some(format!("{stem}-{second}"));
    }
    Some(format!("{stem}{second}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict() -> Dictionary {
        Dictionary::from_words(
            "should\nshall\naddress\nassemble\nabsence\nhimself\nmost\nso\nfo\nfame\nsame\nduring\near\nring\nof\nafter\nMoses\nhistory\nreceive\nfirst\nif\nself\nstaff\nimpatient\ncontinuance"
                .lines(),
        )
    }

    #[test]
    fn knows_inflections_and_names_only_capitalized() {
        let d = dict();
        assert!(d.knows("addressed") && d.knows("receiving") && d.knows("received") && d.knows("rings"));
        assert!(d.knows("Moses") && d.knows("Moses's") && !d.knows("moses"));
        assert!(d.knows("address'd") && d.knows("Most"));
    }

    #[test]
    fn repairs_f_read_for_long_s() {
        let d = dict();
        let fix = |w: &str| repair_f(w, &d);
        assert_eq!(fix("fhould,").as_deref(), Some("should,"));
        assert_eq!(fix("Addreffes").as_deref(), Some("Addresses"));
        assert_eq!(fix("affembled").as_deref(), Some("assembled"));
        assert_eq!(fix("Mofes's").as_deref(), Some("Moses's"));
        assert_eq!(fix("hiftory.").as_deref(), Some("history."));
        assert_eq!(fix("himfelf").as_deref(), Some("himself"));
        assert_eq!(fix("firft").as_deref(), Some("first"));
        assert_eq!(fix("(fo").as_deref(), Some("(so"));
        assert_eq!(fix("Fo"), None); // a capital S is never long
        assert_eq!(fix("fame").as_deref(), Some("same"));
        assert_eq!(fix("faying,").as_deref(), Some("saying,"));
        // real words, a final f, and unknown words stay
        assert_eq!(fix("after"), None);
        assert_eq!(fix("if"), None);
        assert_eq!(fix("staff"), None);
        assert_eq!(fix("Zaphnathpaaneah"), None);
        assert_eq!(fix("xfq"), None);
        assert_eq!(repair_f("perfeft", &Dictionary::from_words(["perse", "perfect"])), None);
        assert_eq!(fix("þ"), None);
        assert_eq!(fix("«hiftoryé»").as_deref(), None);
        assert_eq!(fix("«hiftory»").as_deref(), Some("«history»"));
        // British -our spellings are words, though the list has them as -or
        let d = Dictionary::from_words(["favor", "favorable", "savour", "source", "force"]);
        assert_eq!(repair_f("favour", &d), None);
        assert_eq!(repair_f("Favourable", &d), None);
        assert_eq!(repair_f("favoured,", &d), None);
        assert_eq!(repair_f("fource", &d).as_deref(), Some("source"));
    }

    #[test]
    fn careful_repair_leaves_latin_names_and_fixed_pairs() {
        let d = Dictionary::from_words(["side", "suit", "sol", "same", "last", "shall", "against", "sewer", "fewer", "Rusin", "history", "supposition"]);
        let fix = |w: &str| repair_f_with(w, &d, true, false);
        assert_eq!(fix("fide"), None); // Latin "bona fide"; repair_f takes it for "side"
        assert_eq!(fix("fuit"), None);
        assert_eq!(fix("fol."), None);
        assert_eq!(fix("fame"), None); // fame may be fame: the reader decided
        assert_eq!(fix("Rufin."), None);
        assert_eq!(fix("laft").as_deref(), Some("last")); // f before a consonant
        assert_eq!(fix("fhall").as_deref(), Some("shall"));
        assert_eq!(fix("againft").as_deref(), Some("against"));
        assert_eq!(fix("Hiftory").as_deref(), Some("History"));
        assert_eq!(fix("suppofition").as_deref(), Some("supposition")); // long, lower case
        assert_eq!(repair_f("fide", &d).as_deref(), Some("side"));
        assert_eq!(fix("fewer"), None);
        let d = Dictionary::from_words(["seras", "history"]);
        assert_eq!(repair_f_with("feras", &d, true, true), None); // italic: Latin "feras"
        assert_eq!(repair_f_with("Hiftory", &d, true, true).as_deref(), Some("History"));
    }

    #[test]
    fn fixer_counts_and_respects_mode() {
        let d = dict();
        let mut keep = Fixer::new(LongS::Keep, Some(&d));
        assert_eq!(keep.word("ſhould"), "ſhould");
        let mut s = Fixer::new(LongS::S, Some(&d));
        assert_eq!((s.word("ſhould"), s.word("fhould")), ("should".into(), "fhould".into()));
        let mut repair = Fixer::new(LongS::Repair, Some(&d));
        assert_eq!((repair.word("Moſes"), repair.word("moft")), ("Moses".into(), "most".into()));
        assert_eq!((repair.long_s, repair.repaired), (1, 1));
    }

    #[test]
    fn joins_line_end_hyphens() {
        let d = dict();
        assert_eq!(join_hyphenated("du-", "ring", Some(&d)).as_deref(), Some("during"));
        assert_eq!(join_hyphenated("Continu-", "ance", Some(&d)).as_deref(), Some("Continuance"));
        assert_eq!(join_hyphenated("im-", "patient,", Some(&d)).as_deref(), Some("impatient,"));
        assert_eq!(join_hyphenated("Ear-", "rings", Some(&d)).as_deref(), Some("Ear-rings"));
        assert_eq!(join_hyphenated("Ab-", "fence", Some(&d)).as_deref(), Some("Abfence"));
        assert_eq!(join_hyphenated("New-", "York", Some(&d)), None);
        assert_eq!(join_hyphenated("word", "next", Some(&d)), None);
        assert_eq!(join_hyphenated("du-", "ring", None).as_deref(), Some("during"));
    }
}
