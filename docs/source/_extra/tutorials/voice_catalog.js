"use strict";

// Kokoro-82M voices grouped for the tutorial player's language-first picker.
// `engineCode` is the lang_code passed to KPipeline when narration is rendered.
window.SPACR_VOICE_CATALOG = Object.freeze([
  {
    id: "en",
    label: "English",
    locale: "en",
    voices: [
      { id: "af_heart", name: "Heart", variant: "American female", engineCode: "a" },
      { id: "af_aoede", name: "Aoede", variant: "American female", engineCode: "a" },
      { id: "af_bella", name: "Bella", variant: "American female", engineCode: "a" },
      { id: "af_jessica", name: "Jessica", variant: "American female", engineCode: "a" },
      { id: "af_river", name: "River", variant: "American female", engineCode: "a" },
      { id: "af_sarah", name: "Sarah", variant: "American female", engineCode: "a" },
      { id: "af_sky", name: "Sky", variant: "American female", engineCode: "a" },
      { id: "am_adam", name: "Adam", variant: "American male", engineCode: "a" },
      { id: "am_echo", name: "Echo", variant: "American male", engineCode: "a" },
      { id: "am_eric", name: "Eric", variant: "American male", engineCode: "a" },
      { id: "am_fenrir", name: "Fenrir", variant: "American male", engineCode: "a" },
      { id: "am_liam", name: "Liam", variant: "American male", engineCode: "a" },
      { id: "am_michael", name: "Michael", variant: "American male", engineCode: "a" },
      { id: "am_onyx", name: "Onyx", variant: "American male", engineCode: "a" },
      { id: "am_puck", name: "Puck", variant: "American male", engineCode: "a" },
      { id: "am_santa", name: "Santa", variant: "American male", engineCode: "a" },
      { id: "bf_alice", name: "Alice", variant: "British female", engineCode: "b" },
      { id: "bf_emma", name: "Emma", variant: "British female", engineCode: "b" },
      { id: "bf_isabella", name: "Isabella", variant: "British female", engineCode: "b" },
      { id: "bf_lily", name: "Lily", variant: "British female", engineCode: "b" },
      { id: "bm_daniel", name: "Daniel", variant: "British male", engineCode: "b" },
      { id: "bm_fable", name: "Fable", variant: "British male", engineCode: "b" },
      { id: "bm_george", name: "George", variant: "British male", engineCode: "b" },
      { id: "bm_lewis", name: "Lewis", variant: "British male", engineCode: "b" }
    ]
  },
  {
    id: "es",
    label: "Spanish",
    locale: "es",
    voices: [
      { id: "ef_dora", name: "Dora", variant: "Female", engineCode: "e" },
      { id: "em_alex", name: "Alex", variant: "Male", engineCode: "e" },
      { id: "em_santa", name: "Santa", variant: "Male", engineCode: "e" }
    ]
  },
  {
    id: "fr",
    label: "French",
    locale: "fr-FR",
    voices: [
      { id: "ff_siwis", name: "Siwis", variant: "Female", engineCode: "f" }
    ]
  },
  {
    id: "hi",
    label: "Hindi",
    locale: "hi",
    voices: [
      { id: "hf_alpha", name: "Alpha", variant: "Female", engineCode: "h" },
      { id: "hf_beta", name: "Beta", variant: "Female", engineCode: "h" },
      { id: "hm_omega", name: "Omega", variant: "Male", engineCode: "h" },
      { id: "hm_psi", name: "Psi", variant: "Male", engineCode: "h" }
    ]
  },
  {
    id: "it",
    label: "Italian",
    locale: "it",
    voices: [
      { id: "if_sara", name: "Sara", variant: "Female", engineCode: "i" },
      { id: "im_nicola", name: "Nicola", variant: "Male", engineCode: "i" }
    ]
  },
  {
    id: "pt-BR",
    label: "Brazilian Portuguese",
    locale: "pt-BR",
    voices: [
      { id: "pf_dora", name: "Dora", variant: "Female", engineCode: "p" },
      { id: "pm_alex", name: "Alex", variant: "Male", engineCode: "p" },
      { id: "pm_santa", name: "Santa", variant: "Male", engineCode: "p" }
    ]
  },
  {
    id: "ja",
    label: "Japanese",
    locale: "ja",
    voices: [
      { id: "jf_alpha", name: "Alpha", variant: "Female", engineCode: "j" },
      { id: "jf_gongitsune", name: "Gongitsune", variant: "Female", engineCode: "j" },
      { id: "jf_nezumi", name: "Nezumi", variant: "Female", engineCode: "j" },
      { id: "jf_tebukuro", name: "Tebukuro", variant: "Female", engineCode: "j" },
      { id: "jm_kumo", name: "Kumo", variant: "Male", engineCode: "j" }
    ]
  },
  {
    id: "zh-CN",
    label: "Mandarin Chinese",
    locale: "zh-CN",
    voices: [
      { id: "zf_xiaobei", name: "Xiaobei", variant: "Female", engineCode: "z" },
      { id: "zf_xiaoni", name: "Xiaoni", variant: "Female", engineCode: "z" },
      { id: "zf_xiaoxiao", name: "Xiaoxiao", variant: "Female", engineCode: "z" },
      { id: "zf_xiaoyi", name: "Xiaoyi", variant: "Female", engineCode: "z" },
      { id: "zm_yunjian", name: "Yunjian", variant: "Male", engineCode: "z" },
      { id: "zm_yunxi", name: "Yunxi", variant: "Male", engineCode: "z" },
      { id: "zm_yunxia", name: "Yunxia", variant: "Male", engineCode: "z" },
      { id: "zm_yunyang", name: "Yunyang", variant: "Male", engineCode: "z" }
    ]
  }
]);
