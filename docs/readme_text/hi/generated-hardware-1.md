| Hardware | Cellpose 4 | Torch | UMAP / clustering | 3D |
| --- | --- | --- | --- | --- |
| NVIDIA (CUDA) | 🟢 GPU | 🟢 GPU | 🟢 GPU | 🟣 GPU |
| AMD on Linux (ROCm) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| AMD in an Intel Mac (Metal) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| Apple Silicon (Metal) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| Intel Arc/Xe (XPU) | 🟣 GPU | 🟣 GPU | 🔴 CPU | 🟣 GPU |
| No GPU | 🟢 CPU | 🟢 CPU | 🟢 CPU | 🟢 CPU |

🟢 समर्थित (स्थिर)   🟣 लागू (बीटा)   🔴 केवल CPU समर्थन

Auto: Make Masks का True 3D उपलब्ध बैकएंड इस क्रम में आज़माता है: Linux: Vulkan → OpenGL → CPU; Windows: Vulkan → Direct3D 12 → OpenGL → CPU; macOS: Metal → Vulkan → OpenGL → CPU। Vulkan, Metal और Direct3D 12, wgpu का उपयोग करते हैं। GPU बैकएंड उपलब्ध न होने या विफल होने पर अगला बैकएंड आज़माया जाता है; CPU रेंडरिंग उपलब्ध रहती है।

Windows (x64, ARM64), macOS (Intel, Apple silicon) और Linux (x86-64, ARM64, glibc 2.28 या बाद का संस्करण) पर Python 3.10 या बाद के संस्करण के साथ wgpu डिफ़ॉल्ट रूप से इंस्टॉल होता है। Python 3.9, अन्य प्रोसेसर, पुराने glibc वाले Linux और wgpu के बिना इंस्टॉलेशन पहले OpenGL, फिर CPU का उपयोग करते हैं। बैंगनी 3D सेल लागू किए गए GPU समर्थन को दर्शाते हैं, जिसकी वास्तविक डिस्प्ले पर स्वीकृति जाँच बाकी है।

CPU पर निष्पादन समर्थित है और धीमा हो सकता है। Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB) पर एक माप में torch 2.2.2, Python 3.12.11 और macOS 26.5.2 के साथ एक 256×256 चित्र के लिए Cellpose-SAM ने CPU पर 444.5 s और Metal पर 3.2 s लिए। यह एक पुराना उदाहरण है, गति की गारंटी नहीं। GPU त्वरण के लिए उपयोग योग्य बैकएंड और संगत लाइब्रेरी आवश्यक हैं। [दर्ज किया गया हार्डवेयर माप](<../../../features/new/319_use_non_nvidia_accelerators_where_they_exist.txt>) देखें।
