spaCR, Python <strong>3.9 through 3.14</strong> का समर्थन करता है, Python 3.14.1 को छोड़कर, जिसे torchvision बाहर रखता है। सबसे भारी CUDA और ROCm कार्यप्रवाहों के लिए Linux की अनुशंसा की जाती है; macOS और Windows भी समर्थित हैं, और दोनों अपने GPU का उपयोग करते हैं — macOS, Metal के माध्यम से, जो Apple Silicon और Intel Mac में लगे AMD कार्डों को कवर करता है, और Windows, CUDA या DirectML के माध्यम से।

मानक इंस्टॉलेशन में Qt डेस्कटॉप इंटरफ़ेस शामिल है। सर्वर, क्लस्टर या CI रनर पर इसे खोले बिना कमांड-लाइन पाइपलाइन चलाएँ:
