spaCR styður Python <strong>3.9 til 3.14</strong>, nema Python 3.14.1, sem torchvision útilokar. Mælt er með Linux fyrir þyngstu CUDA- og ROCm-verkflæðin; macOS og Windows eru einnig studd og nýta bæði GPU sín — macOS í gegnum Metal, sem nær yfir Apple Silicon og AMD-kortin í Intel-Mac-tölvum, og Windows í gegnum CUDA eða DirectML.

Hefðbundin uppsetning inniheldur Qt-skjáborðsviðmótið. Á þjóni, reikniklasa eða CI-keyrsluumhverfi má keyra skipanalínuverkflæðin án þess að opna það:
