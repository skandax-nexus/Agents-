#!/usr/bin/env python3
"""Make a multilingual speech corpus with the offline espeak-ng synthesizer (pip install espeakng-loader).

Synthetic formant speech is not a substitute for real recordings, but it carries real phoneme
sequences in many languages, so it is a useful supplement for testing language discovery
when no multilingual recordings are at hand.

    python3 scripts/espeak_corpus.py out_dir --languages en hi es de fr ja --per-language 40
"""
from __future__ import annotations

import argparse
import ctypes
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicecore.audio.io import resample, write_wav  # noqa: E402

SENTENCES = {
    "en": ["The quick brown fox jumps over the lazy dog.", "Please close the window before it rains.",
           "Seven cups of tea were left on the table.", "We will meet at the station at nine.",
           "The river rises every spring after the snow melts.", "Can you hear the birds outside?"],
    "hi": ["नमस्ते, आप कैसे हैं?", "कृपया दरवाज़ा बंद कर दीजिए।", "आज मौसम बहुत अच्छा है।",
           "मुझे चाय पीना पसंद है।", "हम कल बाज़ार जाएँगे।", "नदी गाँव के पास से बहती है।"],
    "es": ["El rápido zorro marrón salta sobre el perro perezoso.", "Por favor, cierra la ventana.",
           "Mañana iremos al mercado temprano.", "El río crece cada primavera.", "¿Puedes oír los pájaros?",
           "Hay siete tazas sobre la mesa."],
    "de": ["Der schnelle braune Fuchs springt über den faulen Hund.", "Bitte schließen Sie das Fenster.",
           "Wir treffen uns um neun am Bahnhof.", "Der Fluss steigt jedes Frühjahr.", "Hörst du die Vögel draußen?",
           "Sieben Tassen stehen auf dem Tisch."],
    "fr": ["Le rapide renard brun saute par-dessus le chien paresseux.", "Veuillez fermer la fenêtre.",
           "Nous irons au marché demain matin.", "La rivière monte chaque printemps.", "Entends-tu les oiseaux dehors ?",
           "Il y a sept tasses sur la table."],
    "ja": ["こんにちは、お元気ですか。", "窓を閉めてください。", "明日、市場に行きます。", "川は春になると増水します。",
           "外で鳥の声が聞こえますか。", "テーブルの上にカップが七つあります。"],
    "ar": ["مرحبا، كيف حالك؟", "من فضلك أغلق النافذة.", "سنذهب إلى السوق غدا.", "يرتفع النهر كل ربيع.",
           "هل تسمع الطيور في الخارج؟", "هناك سبعة أكواب على الطاولة."],
    "sw": ["Habari, hujambo?", "Tafadhali funga dirisha.", "Tutaenda sokoni kesho asubuhi.", "Mto hujaa kila masika.",
           "Unasikia ndege nje?", "Kuna vikombe saba mezani."],
}
VARIANTS = ["", "+m1", "+m3", "+f2", "+f4", "+m5", "+f1"]   # espeak voice variants = different 'speakers'


class ESpeak:
    def __init__(self) -> None:
        import espeakng_loader
        self.lib = ctypes.cdll.LoadLibrary(str(espeakng_loader.get_library_path()))
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.sr = self.lib.espeak_Initialize(1, 500, str(espeakng_loader.get_data_path()).encode(), 0)
        self.samples: list[int] = []
        CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

        def cb(wav, n, events):
            if n > 0 and wav:
                self.samples.extend(wav[:n])
            return 0

        self._cb = CB(cb)
        self.lib.espeak_SetSynthCallback(self._cb)
        self.lib.espeak_SetVoiceByName.restype = ctypes.c_int
        self.lib.espeak_SetParameter.restype = ctypes.c_int

    def say(self, text: str, voice: str, rate: int = 160, pitch: int = 50) -> np.ndarray:
        self.samples.clear()
        self.lib.espeak_SetVoiceByName(voice.encode())
        self.lib.espeak_SetParameter(1, rate, 0)     # espeakRATE
        self.lib.espeak_SetParameter(3, pitch, 0)    # espeakPITCH
        t = text.encode("utf-8")
        self.lib.espeak_Synth(t, len(t) + 1, 0, 0, 0, 0x01, None, None)
        self.lib.espeak_Synchronize()
        return np.array(self.samples, dtype=np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--languages", nargs="+", default=list(SENTENCES))
    ap.add_argument("--per-language", type=int, default=30)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)
    es = ESpeak()
    total = 0.0
    for lang in args.languages:
        sents = SENTENCES.get(lang)
        if not sents:
            print(f"no sentences for {lang}", file=sys.stderr)
            continue
        (out / lang).mkdir(parents=True, exist_ok=True)
        for i in range(args.per_language):
            text = rng.choice(sents)
            variant = rng.choice(VARIANTS)
            wav = es.say(text, lang + variant, rate=rng.randint(140, 185), pitch=rng.randint(35, 70))
            if len(wav) < es.sr // 4:
                continue
            wav = resample(wav, es.sr, args.sr)
            write_wav(out / lang / f"{lang}_{i:03d}.wav", wav, args.sr)
            total += len(wav) / args.sr
        print(f"{lang}: {args.per_language} utterances")
    print(f"wrote {total / 60:.1f} min of synthetic speech to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
