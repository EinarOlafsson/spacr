"""Allowlisted source-coordinate prose audio refresh; no visual/fresh-TTS claim.

Unchanged declared source PCM, including scene holds and gaps, is copied exactly.
Only new English units/changed nonEnglish paragraphs are synthesized/mastered.
Native clocks are bound as immutable native sources, never downgraded to legacy.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,math,os,subprocess,sys
from stage_lesson import REPO,read,write
from check_completed_matrix import digest
import published_scene_audio_refresh as common
RATE=common.RATE;KIND='allowlisted_published_prose_audio';RECORD='published-prose-audio-refresh.json';WORK='published-prose-audio-work'
# Exact reviewed canonical/review content and original selected catalogs/media.
POLICIES={'20_cellpose_masks': {'english_sha256': 'ab0837359e26a864539d5b7388c6b9f5ab14f9d120bd72335ab2baa2ec1c5c99',
                       'source_identity': '5fb52473e46f74f6390c078d0eee88174171a3b1d625289430de0801aeadc44c',
                       'reviews': {'da': '8de34da6971e759c6a8f14069848c3bc966a4f0e3abae14e8d97fe6bdacfc088',
                                   'de': '87aec8b65c83b193a4139a2d9563353ce4cf2a120549cfa7452a9780c7bddaef',
                                   'es': '7f09235b6ce716dfb312b03fea386c49b59ae9d90e8592f10f4a8de3d72c85f0',
                                   'fr': '380bc2c42b7cec08331247fefa4f864decefb640fe976ddfe361724fce09b9f2',
                                   'hi': '98361d6dc9e38499ce6d0ff7c16e8d870766036f82ad1cc8f048fb12a6f140a8',
                                   'is': '4d1e987bb88d04870a412fbab2be08b375a94a6747cc44163c6c1237016639cf',
                                   'it': '1900d3443e888198b4c78fb75b2ad88659ee354bd011ba6c832d7245a0fe6298',
                                   'ja': 'da1122d882cbe8d19a8e6ffe6ed69a3a1b93054fba73e8cb96bcf19a9d6981a9',
                                   'ko': '5a90531277cc3ac7e64fdf809887af763780b16c71642ec27efdbf5d57ff340a',
                                   'nb': '82e3195b620f40c5c080175fda4524df9c80ef346b540552d6cb5ab78da31d53',
                                   'pt-BR': 'a4e37eae1eb62e9d95dbbe10bd50ca74652c5a9ac560ab98743ecccb1c7fcbe0',
                                   'sv': '9cefda66d9414475f35817bb04304ff60b748c199f857e418188f15acecc8fda',
                                   'zh-CN': '9d9a00804d697a95bae2f01d7964d0afdfb2ddeb466cc1c01247077f557bee63'},
                       'voices': {'en': ['af_aoede',
                                         'af_bella',
                                         'af_heart',
                                         'af_jessica',
                                         'af_river',
                                         'af_sarah',
                                         'af_sky',
                                         'am_adam',
                                         'am_echo',
                                         'am_eric',
                                         'am_fenrir',
                                         'am_liam',
                                         'am_michael',
                                         'am_onyx',
                                         'am_puck',
                                         'am_santa',
                                         'bf_alice',
                                         'bf_emma',
                                         'bf_isabella',
                                         'bf_lily',
                                         'bm_daniel',
                                         'bm_fable',
                                         'bm_george',
                                         'bm_lewis'],
                                  'es': ['ef_dora', 'em_alex', 'em_santa'],
                                  'fr': ['ff_siwis'],
                                  'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                  'it': ['if_sara', 'im_nicola'],
                                  'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                  'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                  'zh-CN': ['zf_xiaobei',
                                            'zf_xiaoni',
                                            'zf_xiaoxiao',
                                            'zf_xiaoyi',
                                            'zm_yunjian',
                                            'zm_yunxi',
                                            'zm_yunxia',
                                            'zm_yunyang']},
                       'visual_kind': 'native'},
 '71_investigate_hit': {'english_sha256': '72858a0f292935c05078f9bc0ff092ae845130dd45dda499fb3754a06915b7ec',
                        'source_identity': '8aa4a2c30bb7a0fe2d82ac30e6547d982e5ad992daa004e75ac5e9d4d7159ba1',
                        'reviews': {'da': '5f47d3140364230495022e7ec249824c2d4d48be13a2439eb4eb6ceab83f6b17',
                                    'de': '3bf3996ec6fc0977a036bb3f59abe872970474514b18c901779be8eabd7cdb34',
                                    'es': 'c48294a289a90026f5364d96dd2d9ba11fa8716d740545e7598711cefe6ee359',
                                    'fr': '5cfda7628372edd50d1518e138dbbceaf319cc0295ada92110ad8dd5d26f2e5d',
                                    'hi': '6627b8a43425f172781b49b86797992f889647fbc440a903eb53cbfce243a419',
                                    'is': '8544cc853647f06ece44c458e99c09abc83215bbf1623654a63ef40a0d545659',
                                    'it': '5777b0d1c4019ec8a8427934462a3371816f42bb2575dacc6f245e5cb9e6ab2e',
                                    'ja': '465b234c4e544f1d1843f30d84346763c48b8d14e59b78d5fa45e7f37c264125',
                                    'ko': '6a2f448b53498e46731c2a55efcaad0e85a0ebf5b4a2cbd4f0bdcb39c039699c',
                                    'nb': '5a36698a81142407fa57b83c07487f27b36a4b9facc4cbbbbe0344601583bb06',
                                    'pt-BR': '1213ece85e8cd52018c9f31d226eaa02bea869e64df9c489db43245e734d018e',
                                    'sv': '3c6f29698a2b471e65f250442e607065c46faba99842f74c2e983ef1054a72cc',
                                    'zh-CN': '60e4a37f48dd3cd807582910666849e19f4cbe8fd03671e630abade35011e791'},
                        'voices': {'en': ['af_heart'],
                                   'es': ['ef_dora', 'em_alex', 'em_santa'],
                                   'fr': ['ff_siwis'],
                                   'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                   'it': ['if_sara', 'im_nicola'],
                                   'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                   'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                   'zh-CN': ['zf_xiaobei',
                                             'zf_xiaoni',
                                             'zf_xiaoxiao',
                                             'zf_xiaoyi',
                                             'zm_yunjian',
                                             'zm_yunxi',
                                             'zm_yunxia',
                                             'zm_yunyang']},
                        'visual_kind': 'legacy'},
 '74_import_images': {'english_sha256': '617bf9166de22735ed6d88c416851c82f5bbe413d4d9395b2551a1fde9c9c132',
                      'source_identity': '68b25f3559092e5ca188ab9c89a7aa06c02158aa8eb134049e591a7d42d32ec2',
                      'reviews': {'da': 'c3904d0a1ecd3ee63e86533e890583a5f8bba931918f3697c11f17a52a956abb',
                                  'de': 'd20e48dc3a12e4c602ceae7ccaec69220eefd2f48c1dddc59a73aa9249e010f7',
                                  'es': 'd02ee96530ef4e149c6780ef2671b5559712178a7fc022f150f9171132cf243e',
                                  'fr': '6f76e146be64a5258b2c8d0b7656843ec98c03a802c615ef3174499340e78f89',
                                  'hi': 'd74f4c76e818344a79136c1b1b4be127b7cafc0bdc449c9d1c35dbd2d1992b42',
                                  'is': '36d92960a4f3324c08b9527563183967e48255eb9ced1097240b924aa94720e9',
                                  'it': '2b02450f3d64c6a0169d22d5a00b0be0f5bdf9efd42c3238ad48bb4f3a66ef95',
                                  'ja': 'ef25991f27d30107c8d4b46594b9cb9a7d05e6f5b3686bc657b2991765e19ecc',
                                  'ko': '46b07a47a6d67bca73c50461e067f7006b87156537a52fd4b42c165392517d88',
                                  'nb': 'aa01a53d13c50b4409190e8db4f5fd3f29eab330e11d5e340ac9264b557e1a78',
                                  'pt-BR': '4e51b14c081734212bfafa0af4021bea57854f82d5555b7c906b18e9c0b622ad',
                                  'sv': 'f9ae7657d56adf2211b27aeaac4c8ca6423ac20fea24f207270434848c56c9cc',
                                  'zh-CN': 'ea35ac34a1cea1067055d67c3e5a02a6f598205c00b180607037ff606a63640e'},
                      'voices': {'en': ['af_heart'],
                                 'es': ['ef_dora', 'em_alex', 'em_santa'],
                                 'fr': ['ff_siwis'],
                                 'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                 'it': ['if_sara', 'im_nicola'],
                                 'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                 'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                 'zh-CN': ['zf_xiaobei',
                                           'zf_xiaoni',
                                           'zf_xiaoxiao',
                                           'zf_xiaoyi',
                                           'zm_yunjian',
                                           'zm_yunxi',
                                           'zm_yunxia',
                                           'zm_yunyang']},
                      'visual_kind': 'legacy'},
 '78_spacr_screens': {'english_sha256': '1fe0879fc53974db1818ba1f3adfb18728ce7cb23a1b149e337a7d47f9bfe6a6',
                      'source_identity': '7c8db42659cd140ceff94bff5a57f3dfe704b4270321ef55df937455d20f0be6',
                      'reviews': {'da': '826adcafe90e79955480454347bd97b4a87a40f3bb5280550a580443d303ce93',
                                  'de': '4df3d92d335fd6bb3240b7a2bd98f8c1b003dac452d0602f6942032e344be26e',
                                  'es': '41cb3a8e62594f2549cd0f6bd82097012b9f8cd9b131d34f0f3977181938ef02',
                                  'fr': '436221fa5f5648fa7be4d7970cf63cb82b41669e57584d34ebbad08887195bd5',
                                  'hi': 'b1a861099de4c7fd95c2f1c2f790c0abf7cc1a2d2f74dc92934a6cf532d067e1',
                                  'is': '0b9137ae6ea867f5cb3f27207120d9cb491749fc516faf47cf9a626412b781bf',
                                  'it': 'cd4c2326c139b8d270bbeccd394254fb48af15cfddccdedfd7b3aef87fc04852',
                                  'ja': '99bfde555d1c728795f033627f2c316c0ddbf5c8c4cc95b06a08da78d5d7f4d9',
                                  'ko': '9e263a96c2f16787891a07452e65307ec047146ab5c2d8c3cfd4ec88a5da6d7c',
                                  'nb': '8e6d87e7bacc73b166a291d665db8ec3d1b66fbb165f778431b9d17ac236142c',
                                  'pt-BR': '9cc577ad56ef4f78da55f27b26d884fc7143847d8de8e8e11439237cf8c0234e',
                                  'sv': '374e4a2272f508c1e22d50664a6adf7200fc7d3aadda0a326a253c4a8ec96f69',
                                  'zh-CN': '3be64abf36bed460a53ff98e57273513027729145f02d661bd83fd283227a43b'},
                      'voices': {'en': ['af_aoede',
                                        'af_bella',
                                        'af_heart',
                                        'af_jessica',
                                        'af_river',
                                        'af_sarah',
                                        'af_sky',
                                        'am_adam',
                                        'am_echo',
                                        'am_eric',
                                        'am_fenrir',
                                        'am_liam',
                                        'am_michael',
                                        'am_onyx',
                                        'am_puck',
                                        'am_santa',
                                        'bf_alice',
                                        'bf_emma',
                                        'bf_isabella',
                                        'bf_lily',
                                        'bm_daniel',
                                        'bm_fable',
                                        'bm_george',
                                        'bm_lewis'],
                                 'es': ['ef_dora', 'em_alex', 'em_santa'],
                                 'fr': ['ff_siwis'],
                                 'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                 'it': ['if_sara', 'im_nicola'],
                                 'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                 'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                 'zh-CN': ['zf_xiaobei',
                                           'zf_xiaoni',
                                           'zf_xiaoxiao',
                                           'zf_xiaoyi',
                                           'zm_yunjian',
                                           'zm_yunxi',
                                           'zm_yunxia',
                                           'zm_yunyang']},
                      'visual_kind': 'native'},
 '80_image_analysis_pathways': {'english_sha256': 'ea604d1f023bbbd1210df4544feebc683f1d71559886adbbf265ee16392f7fbb',
                                'source_identity': '18f522e233c4fc3c2cb061caa206410be155f7edd5b15faf23b3021d7e08d993',
                                'reviews': {'da': '506cdc82fdf33f19f3d93163383060545a45f30589e569a8d49d79700fd7c067',
                                            'de': '3b87fb8d8c85031e0147aeedc5c6a10d707a14dc8a060403be6d5e831604e186',
                                            'es': 'a0c4ca96933a91fb228e24e10487137562651a0a2b42dbdd95e138b1b8be80df',
                                            'fr': '6789f714f2c237a9a5402cd633d69d9160ac0f7864e97a24b5eedc094dfadac5',
                                            'hi': '0c3b553adddd7b4395908dcddb3076973128dc391f8a93fa3a20252dd6bce87a',
                                            'is': '7e9cc8097cd9b40b74870558c64d42e11b9193518833573c2b96061cc9dfca23',
                                            'it': '45c135bcd17f22d1c788713ff2d114ded9362e15c2df4d43eae638fd78b38b2a',
                                            'ja': '3689f1ea7064e8984cb2ab0c2b5031939e155502e762e052ead1b26f1b9eb8a1',
                                            'ko': '384376ce38c2ce409544cf72269b0d127f3d58ffefc35f3babbac9e8dbaddb44',
                                            'nb': '9a8601b0326b00ae6d498ff40d3461b324bfd1b81acf00f003e9938bce255b98',
                                            'pt-BR': 'fa88513fceaedea0b502e2a31e51cc125c68d3dc048f14cf0a05144d4a901e1c',
                                            'sv': 'b00aee2b1b056f739587db8c838fdab72e824d480efe7befeba0f11c6dc50f6f',
                                            'zh-CN': '83206e2d340b2b4b3e4af381e71105ca97ca2069156586dc5bf627f0cba41971'},
                                'voices': {'en': ['af_aoede',
                                                  'af_bella',
                                                  'af_heart',
                                                  'af_jessica',
                                                  'af_river',
                                                  'af_sarah',
                                                  'af_sky',
                                                  'am_adam',
                                                  'am_echo',
                                                  'am_eric',
                                                  'am_fenrir',
                                                  'am_liam',
                                                  'am_michael',
                                                  'am_onyx',
                                                  'am_puck',
                                                  'am_santa',
                                                  'bf_alice',
                                                  'bf_emma',
                                                  'bf_isabella',
                                                  'bf_lily',
                                                  'bm_daniel',
                                                  'bm_fable',
                                                  'bm_george',
                                                  'bm_lewis'],
                                           'es': ['ef_dora', 'em_alex', 'em_santa'],
                                           'fr': ['ff_siwis'],
                                           'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                           'it': ['if_sara', 'im_nicola'],
                                           'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                           'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                           'zh-CN': ['zf_xiaobei',
                                                     'zf_xiaoni',
                                                     'zf_xiaoxiao',
                                                     'zf_xiaoyi',
                                                     'zm_yunjian',
                                                     'zm_yunxi',
                                                     'zm_yunxia',
                                                     'zm_yunyang']},
                                'visual_kind': 'native'},
 '81_sequencing_pathways': {'english_sha256': '8f32b8c4f1b3064947ce7778a6b20c6beb16457afa8ae0794a3ba4fb990ca7ee',
                            'source_identity': '589be7ca6446c3f16efa83ad5df816fd48066c1ac8a24a878caf35f01f422076',
                            'reviews': {'da': 'a2847d2cee72d94cfc8d0a511ad86f88240ed6376a8c084d579eb7ab56b83419',
                                        'de': '4b4aecdc83bd11fdde927e41e7a30d3ea147417f1687f4e2f769e246aea29e12',
                                        'es': '0c3e23220e9a54c9645a03c071d7943d9be81c7fd87170f15efafac3590e3b9e',
                                        'fr': '7c213e5ad158e1c189175f2107a46783ba3cb953bc60f4f8ec5c5a49898e2494',
                                        'hi': 'de530409e466d9238428ebc01faf9b737b102ba6982313f160dddb256189ba25',
                                        'is': 'ec6d3548edeaa926063c00660792dbaa3b9b774f67abaffaede8e596810ad2ea',
                                        'it': 'b46ee94f7caefe8a65f44d55d1a007e044e16800cecdd1b4d29d50f669dbb2a7',
                                        'ja': '6ed84e74a49a7d32bd50537a5252363be5e21226b68e3d2d442c499b27b68889',
                                        'ko': '8d4a5e2ce82f28e05cf7eb2b28ce55f83a4defdc225c54a468428cf60bb254ee',
                                        'nb': '4f1b8ff8709475c77e9c522bf114191860e208fe238d15659cdc8856e6a60766',
                                        'pt-BR': '9a15f1ade13c4defbc39b455487cd3f16c32966aa681903caedc9b62d91a4cff',
                                        'sv': '345c67d9f28f08b5273ca5e974839afb1150488769c814d1135e4bd60507b34a',
                                        'zh-CN': '0e75000c190882a549467ae9ba880c8592fa83ccbaa44e3b658dbc983b484ffc'},
                            'voices': {'en': ['af_heart'],
                                       'es': ['ef_dora', 'em_alex', 'em_santa'],
                                       'fr': ['ff_siwis'],
                                       'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                       'it': ['if_sara', 'im_nicola'],
                                       'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                       'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                       'zh-CN': ['zf_xiaobei',
                                                 'zf_xiaoni',
                                                 'zf_xiaoxiao',
                                                 'zf_xiaoyi',
                                                 'zm_yunjian',
                                                 'zm_yunxi',
                                                 'zm_yunxia',
                                                 'zm_yunyang']},
                            'visual_kind': 'legacy'},
 '86_alpha_organism_modules': {'english_sha256': 'c5e9f20368aa0fb565d3d1d48ed1209e4794abf585615c387db512cdd3cb80b2',
                               'source_identity': 'e05b8eb788f9e4c32625fecef2b2ee6153792489ce9dab06b2b547f77b4e4ea5',
                               'reviews': {'da': 'e4724caf13d2121d2763ddcf494bd663bdcdc25ff46d1ac36f6135e81eda6924',
                                           'de': '95bd6fa03331ee74546d26d83e73d1638f0108b1c6e8f6b02ac21a4ce72bb496',
                                           'es': 'b57d287fffff58c1cfbf6751533402022ca5eea41dd52d399c83f5783d21a72f',
                                           'fr': 'ef18687c24bda7612a2b4e7447e7b5b2ef7dd0c10f43be71aacd206c8147d477',
                                           'hi': '693f459164930d05178da849536583feedcc8029a49044c547183aa5078a72d3',
                                           'is': 'a3f3323061ed713e3157ab4fa8cacc6de3f7624372377dc0b60f9d5d76a99c07',
                                           'it': '41ffaa1e5e4ce83a156e84b275a15de4b561642a6b1407791624c7ab0c71bec1',
                                           'ja': '14aa2a347d41518ea2d53db02c53d7ad20046f33bcf14a2982acaf36e4b88d0a',
                                           'ko': '727d3e667a81d88bcf4a3185ccd7867b81088f21aee54d8d280dc58e3742bd47',
                                           'nb': '2d09be16732d4aa07224806ccf2e3bfb8dc1eea231db98493a00978e9d36c9e9',
                                           'pt-BR': 'eac409a3660c1cbd1ad9c6c4b4a2abacb78d71ed72abdafbfe69ee29c3125353',
                                           'sv': '198069ad8bfcace7db354e9001276b416fe82e772bf3651fefa5c55a88bffbaf',
                                           'zh-CN': 'de01b5e191633aaa1fddd3b9846231a16ce762731f38675fd6a89b5a524bc6d2'},
                               'voices': {'en': ['af_heart'],
                                          'es': ['ef_dora', 'em_alex', 'em_santa'],
                                          'fr': ['ff_siwis'],
                                          'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                          'it': ['if_sara', 'im_nicola'],
                                          'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                          'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                          'zh-CN': ['zf_xiaobei',
                                                    'zf_xiaoni',
                                                    'zf_xiaoxiao',
                                                    'zf_xiaoyi',
                                                    'zm_yunjian',
                                                    'zm_yunxi',
                                                    'zm_yunxia',
                                                    'zm_yunyang']},
                               'visual_kind': 'legacy'},
 '87_alpha_features': {'english_sha256': 'c666c5555666207635f969ea34f8456751fda8dad105a37d956ebb4592b98a38',
                       'source_identity': '09cd281b97997ba63b1ce648f7ecc777edc343b22954604ccc4da813990c8eff',
                       'reviews': {'da': '42412ed09c1402be49093f1cf7983c732514fbf230f86cdd8c0a24f2dc289a04',
                                   'de': '0bccfedaf4d7bf99122381f0a23044cc9a5b62af4b539cca2d4e7484233e0678',
                                   'es': 'eedd4ed72a405d938c9787c499787a6b9771c64ef40bd76bf3bc8e50a817aaea',
                                   'fr': '1b3ca6a05ae968945234eb8da234d4639954d0e91a0b244a2e992036d10fbeed',
                                   'hi': 'cbe0344b83e04d0c71a22366a7fa58badca93e2de1c521393de3b9e80e407679',
                                   'is': '851e219314f7526707a1de50ad46fa417bb6ad420fd0779814b03be5c0c4884c',
                                   'it': '1e95dcb1a4bf866e091ffb18a84ec224ab1db9c2d8dbe4e5c3206479ee21ec7e',
                                   'ja': '61d8bb2ad9925ddc26e2adef8441600d54b65ba10aac22aee78a3891c1b063b7',
                                   'ko': 'ca34e5451b5db1d003f6c88e76818ceb09924b8de4e8a20714352e63540fb28b',
                                   'nb': '2ee6a88f32432707a9855d2ee860937bdd6cdeec4d7d3c3f3eb951d6e0110206',
                                   'pt-BR': 'e56ceea0e7f8474d23533c499fb530816e50019d02a065261e466eb53a5fec65',
                                   'sv': 'a70e586f9fab8970179c572fed4c550b0ace94c747784b748129a6ee42b64691',
                                   'zh-CN': '4580c23c979a64aaaaa02e5061c3294d02a70b4d28ebe264404bba84b2f78dea'},
                       'voices': {'en': ['af_heart'],
                                  'es': ['ef_dora', 'em_alex', 'em_santa'],
                                  'fr': ['ff_siwis'],
                                  'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
                                  'it': ['if_sara', 'im_nicola'],
                                  'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
                                  'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
                                  'zh-CN': ['zf_xiaobei',
                                            'zf_xiaoni',
                                            'zf_xiaoxiao',
                                            'zf_xiaoyi',
                                            'zm_yunjian',
                                            'zm_yunxi',
                                            'zm_yunxia',
                                            'zm_yunyang']},
                       'visual_kind': 'legacy'}}

def selected(catalog,lesson):
 rows=[r for r in catalog['lessons'] if r.get('id')==lesson]
 if len(rows)!=1:raise ValueError('Exactly one selected lesson required')
 return rows[0]

def selection(lessons):
 if not lessons or len(lessons)!=len(set(lessons)) or set(lessons)-POLICIES.keys():raise ValueError('Select unique allowlisted lessons;77 is held')
 return list(lessons)

def render_selection(lessons):
 lessons=selection(lessons)
 if len(lessons)!=1:raise ValueError('Render exactly one approved lesson per invocation')
 return lessons

def require_original(policy,catalogs,media):
 if common.identity(dict(catalogs=catalogs,media=sorted(media,key=lambda r:r['path'])))!=policy['source_identity']:raise ValueError('Foreign original selected catalog/media source')


def native_source(baseline,lesson,english):
 """New wrapper around actual immutable published-native helpers only."""
 from published_native_reuse import _published_inputs,interval_spec
 baseline,native,bindings,source=_published_inputs(baseline,lesson)
 interval_spec(english,native)
 from append_staged_lessons import parse_javascript
 row=selected(parse_javascript((baseline/'web/lesson_catalog.js').read_text()),lesson)
 if row.get('visual_timings')!=bindings['timing']['path'].removeprefix('media_host/'):raise ValueError('Native runtime/source clock differs')
 return dict(kind='native_visual_timing',path=bindings['timing']['path'],sha256=bindings['timing']['sha256'],published_bindings=bindings),native

def sources(stage,baseline,lessons):
 from audit_staged_catalogs import CATALOGS
 from build_audio_patch_candidate import require_published_script_baseline,legacy_visual_source
 from published_retained_narration import _authoring_without_generated_aliases
 from apply_translation_review import translated_lesson
 from retain_narration import require_timing
 lessons=selection(lessons);stage,baseline=Path(stage),Path(baseline)
 if stage.is_symlink() or baseline.is_symlink():raise ValueError('Regular independent roots required')
 stage,baseline=stage.resolve(),baseline.resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Independent stage required')
 manifest=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,manifest);records={r['path']:r for r in manifest['files']}
 if len(records)!=len(manifest['files']):raise ValueError('Duplicate published paths')
 def bound(relative):
  p=common.regular(baseline,relative);r=records.get(relative)
  if not r or (digest(p),p.stat().st_size)!=(r['sha256'],r['bytes']):raise ValueError('Published source changed: '+relative)
  return p
 old={};current={};reviews={l:{} for l in lessons}
 for name in CATALOGS:
  a=read(bound('web/catalog/'+name));b=read(common.regular(stage,'catalog/'+name));ids=[r['id'] for r in a['lessons']]
  if len(ids)!=len(set(ids)) or ids!=[r['id'] for r in b['lessons']] or any(ids.count(l)!=1 for l in lessons):raise ValueError('Catalog inventory/order differs')
  expected=deepcopy(a)
  for lesson in lessons:expected['lessons'][ids.index(lesson)]=selected(b,lesson)
  if b!=expected:raise ValueError('Only explicitly selected catalog objects may change')
  old[name],current[name]=a,b
 answer={}
 for lesson in lessons:
  policy=POLICIES[lesson];root=stage/'production'/lesson;en=selected(current['lessons_en.json'],lesson);canonical=read(REPO/f'tools/tutorials/lessons/{lesson}.json');staged=read(common.regular(stage,f'production/{lesson}/lesson.en.json'))
  if common.identity(canonical)!=policy['english_sha256'] or staged!=canonical or _authoring_without_generated_aliases({k:v for k,v in en.items() if k!='narration_voices'},canonical,'en')!=canonical:raise ValueError('Only reviewed exact canonical/current English')
  originals={name:selected(c,lesson) for name,c in old.items()};media=sorted([r for r in manifest['files'] if r['path'].startswith('media_host/'+lesson+'/') or r['path']==f'web/production/{lesson}/poster.jpg'],key=lambda r:r['path'])
  require_original(policy,originals,media)
  for name in CATALOGS:
   if name=='lessons_en.json':continue
   lang=name.split('_',1)[1].removesuffix('.json');p=common.regular(stage,f'production/{lesson}/review.{lang}.json');r=read(p)
   if common.identity({k:v for k,v in r.items() if k!='review'})!=policy['reviews'][lang] or p.read_bytes()!=(REPO/f'tools/tutorials/lessons/reviews/{lesson}.{lang}.json').read_bytes() or not r.get('review') or translated_lesson(r,en,originals[name])!=selected(current[name],lesson):raise ValueError('Exact adopted reviewed localized projection required')
   reviews[lesson][lang]=digest(p)
  voices=originals['lessons_en.json']['narration_voices']
  if voices!=policy['voices'] or en.get('narration_voices')!=voices:raise ValueError('Exact original offering matrix required')
  if policy['visual_kind']=='native':
   visual,native=native_source(baseline,lesson,originals['lessons_en.json']);proofs=native.get('source_tracks',[]);proof_by={(p.get('language'),p.get('voice')):p for p in proofs}
   if len(proofs)!=50 or len(proof_by)!=50:raise ValueError('Exact fifty original native source proofs required')
  else:
   clock=legacy_visual_source(baseline,lesson,originals['lessons_en.json']);visual=dict(kind='legacy',path=clock.relative_to(baseline).as_posix(),sha256=digest(clock));proof_by={}
  bound(visual['path']);prefix=f'media_host/{lesson}/audio/';expected={prefix+f'{l}/{v}{suffix}' for l,vs in voices.items() for v in vs for suffix in ('.m4a','.json')}
  if {p for p in records if p.startswith(prefix)}!=expected:raise ValueError('Original audio inventory differs')
  tracks=[]
  for lang,offered in voices.items():
   for voice in offered:
    a=bound(prefix+f'{lang}/{voice}.m4a');t=bound(prefix+f'{lang}/{voice}.json');timing=read(t);require_timing(timing,originals['lessons_'+lang+'.json'],lang,voice,digest(a))
    if policy['visual_kind']=='native':
     inputs=timing['render_inputs'];fp=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest();proof=dict(language=lang,voice=voice,audio_sha256=digest(a),timing_sha256=digest(t),render_fingerprint=fp)
     if timing.get('render_fingerprint')!=fp or proof_by.get((lang,voice))!=proof:raise ValueError('Original native proof/fingerprint differs')
    tracks.append(dict(language=lang,voice=voice,audio=a.relative_to(baseline).as_posix(),timing=t.relative_to(baseline).as_posix(),audio_sha256=digest(a),timing_sha256=digest(t)))
  answer[lesson]=dict(schema=1,kind=KIND,lesson=lesson,selected_lessons=lessons,baseline_root=str(baseline),baseline_manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),source_identity=policy['source_identity'],current_english_sha256=policy['english_sha256'],catalogs={n:common.identity(selected(c,lesson)) for n,c in current.items()},reviews=reviews[lesson],voices=voices,original_visual_clock=visual,native_adapter_required=policy['visual_kind']=='native',tracks=tracks,fresh_capture=False,current_whole_track_tts_claimed=False,human_listening_review=False)
 return answer,current

def sample(value):
 if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite source time required')
 return round(value*RATE)

def plan_track(plans,old,language):
 """Only exact, unique, monotonic within-index unit anchors; no fuzzy matching."""
 if len(plans)!=len(old.get('scenes',[])) or not plans:raise ValueError('Indexed scene inventory differs')
 ops=[];scene_plans=[];previous=0
 for index,(plan,scene) in enumerate(zip(plans,old['scenes']),1):
  a,end,b=map(sample,(scene['speech_start'],scene['speech_end'],scene['scene_end']))
  if type(scene.get('scene')) is not int or scene['scene']!=index or a!=previous or not a<end<=b:raise ValueError('Contiguous original source scenes required')
  previous=b;units=scene['sentences'];cursor=a
  for j,u in enumerate(units,1):
   x,y=sample(u['speech_start']),sample(u['speech_end'])
   if u.get('sentence')!=j or not cursor<=x<y<=end or not x<=sample(u['audible_start'])<sample(u['audible_end'])<=y:raise ValueError('Original indexed unit bounds differ')
   cursor=y
  if not units or sample(units[0]['speech_start'])!=a or sample(units[-1]['speech_end'])!=end:raise ValueError('Original speech units must cover original speech clip')
  changed=plan['display_text']!=scene['text']
  # Exact unchanged scene is copied whole, regardless of current runtime/cache.
  if not changed:scene_plans.append(dict(scene=index,changed=False,anchors=[]));continue
  if plan['effective_speed']!=scene['effective_speed']:raise ValueError('Changed-unit speed requires separate review')
  new=plan['sentences'];anchors=[]
  if not new:raise ValueError('Current nonempty speech units required')
  if language=='en':
   from narration_audio import split_sentences
   # Unchanged display material must retain its exact published spoken unit.
   # A new alias or different unit grouping needs a reviewed indexed mapping.
   unchanged_display=set(split_sentences(scene['text'],'en')) & set(split_sentences(plan['display_text'],'en'))
   unchanged_display.update(u['text'] for u in new if any(v['text']==u['text'] for v in units))
   for text in unchanged_display:
    before=[k for k,v in enumerate(units) if v['text']==text];after=[j for j,u in enumerate(new) if u['text']==text]
    if len(before)!=1 or len(after)!=1 or units[before[0]]['speech_text']!=new[after[0]]['speech_text']:raise ValueError('Unchanged display unit requires exact published spoken anchor; review indexed mapping')
   for j,u in enumerate(new):
    matches=[k for k,v in enumerate(units) if v['speech_text']==u['speech_text']]
    if len(matches)>1 or matches and sum(v['speech_text']==u['speech_text'] for v in new)>1:raise ValueError('Ambiguous duplicate unit requires reviewed explicit mapping')
    if matches:anchors.append((matches[0],j))
   if any(x[0]>=y[0] for x,y in zip(anchors,anchors[1:])):raise ValueError('Unit reorder requires separate reviewed plan')
  elif len(new)!=1 or len(units)!=1:raise ValueError('NonEnglish uses exact whole-paragraph units')
  elif new[0]['speech_text']==units[0]['speech_text']:anchors=[(0,0)]
  scene_plans.append(dict(scene=index,changed=True,anchors=[list(x) for x in anchors]))
  bounds=[(-1,-1)]+anchors+[(len(units),len(new))]
  for (old_left,new_left),(old_right,new_right) in zip(bounds,bounds[1:]):
   old_start,old_stop=old_left+1,old_right;new_start,new_stop=new_left+1,new_right
   if old_start==old_stop and new_start==new_stop:continue
   if old_start<old_stop:x,y=sample(units[old_start]['speech_start']),sample(units[old_stop-1]['speech_end'])
   else:x=y=sample(units[old_right]['speech_start']) if old_right<len(units) else end
   # Insertion preserves the original neighboring gap; supply only the missing
   # new-to-next pause, or previous-to-new pause for end insertion.
   prepend=old_start==old_stop and old_right==len(units) and new_start<new_stop
   append=old_start==old_stop and old_right<len(units) and new_start<new_stop
   ops.append(dict(scene=index,source_samples=[x,y],old_units=[old_start,old_stop],new_units=[new_start,new_stop],prepend_pause=prepend,append_pause=append))
 if any(x['source_samples'][1]>y['source_samples'][0] for x,y in zip(ops,ops[1:])):raise ValueError('Overlapping source replacements')
 return dict(operations=ops,scenes=scene_plans,declared_samples=previous)

def splice(source,operations,replacements,declared,np):
 """Preserve every source-complement sample; ranges cover the entire declaration."""
 if source.dtype!=np.dtype('<i2') or source.ndim!=1 or type(declared) is not int or not 0<declared<=len(source) or len(operations)!=len(replacements):raise ValueError('Declared mono source and replacement inventory required')
 parts=[];ranges=[];cursor=output=0
 for op,pcm in zip(operations,replacements):
  a,b=op['source_samples']
  if any(type(v) is not int for v in (a,b)) or not cursor<=a<=b<=declared or pcm.dtype!=np.dtype('<i2') or pcm.ndim!=1:raise ValueError('Ordered bounded source-coordinate replacements required')
  if cursor<a:
   old=source[cursor:a];parts.append(old);ranges.append(dict(kind='published_source_complement',source_samples=[cursor,a],output_samples=[output,output+len(old)],pcm_sha256=hashlib.sha256(old.tobytes()).hexdigest()));output+=len(old)
  parts.append(pcm);ranges.append(dict(kind='changed_replacement',scene=op['scene'],source_samples=[a,b],output_samples=[output,output+len(pcm)],pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()));output+=len(pcm);cursor=b
 if cursor<declared:
  old=source[cursor:declared];parts.append(old);ranges.append(dict(kind='published_source_complement',source_samples=[cursor,declared],output_samples=[output,output+len(old)],pcm_sha256=hashlib.sha256(old.tobytes()).hexdigest()))
 return np.concatenate(parts).astype('<i2',copy=False),ranges

def point(x,operations,replacements,include_equal=lambda op:False):
 offset=0
 for op,pcm in zip(operations,replacements):
  a,b=op['source_samples']
  if a==b:
   if x>a or x==a and include_equal(op):offset+=len(pcm)
  elif x>=b:offset+=len(pcm)-(b-a)
  elif a<x<b:raise ValueError('Retained metadata falls inside removed speech')
 return x+offset

def replacement_block(renderer,plan,units,raws,phonemes,op,np):
 from narration_audio import SENTENCE_PAUSE_SECONDS
 if not units:return np.zeros(0,dtype='<i2'),[]
 speech,segments=renderer.assemble_sentence_audio(raws,np);peak=float(np.max(np.abs(speech)))
 if peak>.98:speech=speech*(.98/peak)
 pause=round(SENTENCE_PAUSE_SECONDS*RATE);lead=pause if op['prepend_pause'] else 0;tail=pause if op['append_pause'] else 0
 rows=[]
 for unit,segment,phones in zip(units,segments,phonemes):
  rows.append(dict(sentence=0,speech_start=segment['clip_start']+lead/RATE,speech_end=segment['clip_end']+lead/RATE,audible_start=segment['audible_start']+lead/RATE,audible_end=segment['audible_end']+lead/RATE,duration=segment['duration'],trimmed_lead=segment['trimmed_lead'],trimmed_tail=segment['trimmed_tail'],gap_from_previous=None,**unit,phonemes=phones))
 full=np.concatenate([np.zeros(lead,dtype=np.float32),speech,np.zeros(tail,dtype=np.float32)])
 return np.clip(np.floor(full*32768),-32768,32767).astype('<i2'),rows

def master_changed(pcm,folder,renderer,lesson,language,voice,np):
 from narration_audio import ffmpeg_runtime_identity
 config=renderer.mastering_config(lesson,language,voice);chosen=config['filters'][0]
 if chosen!='loudnorm=I=-16:TP=-3:LRA=11':raise ValueError('Only normal first replacement mastering')
 a,b=folder/'unmastered.s16le',folder/'mastered.s16le'
 if any(os.path.lexists(p) for p in (a,b)):raise FileExistsError('Preserve replacement mastering')
 a.write_bytes(pcm.tobytes());subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(a),'-af',chosen,'-ar',str(RATE),'-ac','1','-c:a','pcm_s16le','-f','s16le',str(b)],check=True)
 if b.stat().st_size!=len(pcm)*2:raise ValueError('Replacement mastering changed sample count')
 return np.frombuffer(b.read_bytes(),dtype='<i2'),dict(configuration=config,filter=chosen,tool=ffmpeg_runtime_identity(),unmastered_sha256=digest(a),mastered_sha256=digest(b),samples=len(pcm))

def extension_specs(old,scope):
 """Only edge-deletion orphan gaps between complete old unit clips may extend metadata."""
 result=[]
 for row in scope['scenes']:
  if not row['changed'] or not row['anchors']:continue
  i=row['scene'];units=old['scenes'][i-1]['sentences'];first,last=row['anchors'][0],row['anchors'][-1]
  scene_ops=[op for op in scope['operations'] if op['scene']==i];new_count=max([j+1 for _,j in row['anchors']]+[op['new_units'][1] for op in scene_ops])
  if first[1]==0 and first[0]>0:
   k=first[0];a,b=sample(units[k-1]['speech_end']),sample(units[k]['speech_start'])
   if a<b:result.append(dict(scene=i,side='lead',source_samples=[a,b],left_unit=k-1,right_unit=k))
  if last[1]==new_count-1 and last[0]<len(units)-1:
   k=last[0];a,b=sample(units[k]['speech_end']),sample(units[k+1]['speech_start'])
   if a<b:result.append(dict(scene=i,side='tail',source_samples=[a,b],left_unit=k,right_unit=k+1))
 return result


def prove_extensions(source,old,scope,np):
 """Exact old VAD bounds plus normal per-unit PCM threshold; codec noise may remain."""
 from narration_audio import trim_edge_silence,audio_assembly_config,VAD_ABSOLUTE_FLOOR,VAD_FRAME_SECONDS
 proofs=[]
 for spec in extension_specs(old,scope):
  a,b=spec['source_samples'];units=old['scenes'][spec['scene']-1]['sentences'];left,right=units[spec['left_unit']],units[spec['right_unit']]
  if not sample(left['audible_end'])<=a<b<=sample(right['audible_start']):raise ValueError('Metadata extension overlaps original declared speech')
  thresholds=[]
  for unit in (left,right):
   clip=source[sample(unit['speech_start']):sample(unit['speech_end'])].astype(np.float32)/32768
   _,info=trim_edge_silence(clip,np,RATE);thresholds.append(info.get('threshold',VAD_ABSOLUTE_FLOOR))
  # Normal unit-specific thresholds on both adjacent clips: quiet must meet
  # even the more conservative threshold, without inventing a codec allowance.
  threshold=min(thresholds);gap=source[a:b].astype(np.float32)/32768;size=max(1,round(VAD_FRAME_SECONDS*RATE))
  peak=max(float(np.max(np.abs(gap[j:j+size]))) for j in range(0,len(gap),size))
  if peak>=threshold:raise ValueError('Unproven silent gap requires explicit reviewed silence-only plan')
  proofs.append(dict(**spec,normal_assembly=audio_assembly_config(),threshold=threshold,maximum_frame_peak=peak,pcm_sha256=hashlib.sha256(source[a:b].tobytes()).hexdigest()))
 return proofs


def retained_proofs(source,joined,old,scope,replacements,np):
 proofs=[];ops=scope['operations']
 for row in scope['scenes']:
  i=row['scene'];units=old['scenes'][i-1]['sentences'];anchors=row['anchors'] if row['changed'] else [(j,j) for j in range(len(units))]
  for k,j in anchors:
   u=units[k];a,b=sample(u['speech_start']),sample(u['speech_end']);x=point(a,ops,replacements,lambda op:True);y=point(b,ops,replacements)
   if joined[x:y].tobytes()!=source[a:b].tobytes():raise ValueError('Original retained unit PCM changed')
   proofs.append(dict(scene=i,old_sentence=k+1,new_sentence=j+1,source_samples=[a,b],output_samples=[x,y],pcm_sha256=hashlib.sha256(source[a:b].tobytes()).hexdigest()))
 return proofs


def declared_sound_gaps(scenes):
 units=[u for scene in scenes for u in scene['sentences']];checks=[]
 for left,right in zip(units,units[1:]):
  gap=right['audible_start']-left['audible_end']
  if not math.isfinite(gap) or gap<0 or gap>1+1/RATE:raise ValueError('Declared sound gap exceeds1s; explicit reviewed silence-only cut required')
  checks.append(dict(last_sound=left['audible_end'],next_sound=right['audible_start'],seconds=gap))
 if not units or units[0]['audible_start']>1+1/RATE or scenes[-1]['scene_end']-units[-1]['audible_end']>1+1/RATE:raise ValueError('Leading/trailing silence exceeds1s; explicit reviewed silence-only plan required')
 return checks


def current_scenes(plans,old,plan,replacements,replacement_rows,extension_proofs=()):
 ops=plan['operations'];result=[]
 for i,(p,scene,scope) in enumerate(zip(plans,old['scenes'],plan['scenes']),1):
  row=deepcopy(scene)
  for key in ('speech_start','speech_end','scene_end'):
   include=(lambda op:op['scene']<=i) if key!='speech_start' else (lambda op:op['scene']<i)
   row[key]=point(sample(scene[key]),ops,replacements,include)/RATE
  row['duration']=row['scene_end']-row['speech_start']
  if scope['changed']:
   units=[None]*len(p['sentences'])
   for old_index,new_index in scope['anchors']:
    u=deepcopy(scene['sentences'][old_index])
    for key in ('speech_start','speech_end','audible_start','audible_end'):
     u[key]=point(sample(u[key]),ops,replacements,(lambda op:True) if key in ('speech_start','audible_start') else (lambda op:False))/RATE
    u.update(p['sentences'][new_index]);units[new_index]=u
   for n,op in enumerate(ops):
    if op['scene']!=i:continue
    # The operation's output starts after every preceding operation, including
    # any earlier insertion at the same source coordinate.
    start=op['source_samples'][0]+sum(len(pcm)-(prev['source_samples'][1]-prev['source_samples'][0]) for prev,pcm in zip(ops[:n],replacements[:n]))
    for j,u in zip(range(*op['new_units']),replacement_rows[n]):
     u=deepcopy(u)
     for key in ('speech_start','speech_end','audible_start','audible_end'):u[key]+=start/RATE
     units[j]=u
   if any(u is None for u in units):raise ValueError('Every current unit must be accounted for')
   for side,unit,key,target in [('lead',units[0],'speech_start',row['speech_start']),('tail',units[-1],'speech_end',row['speech_end'])]:
    if abs(unit[key]-target)>1/RATE:
     matches=[proof for proof in extension_proofs if proof['scene']==i and proof['side']==side]
     if len(matches)!=1 or matches[0]['maximum_frame_peak']>=matches[0]['threshold'] or abs(abs(unit[key]-target)-(matches[0]['source_samples'][1]-matches[0]['source_samples'][0])/RATE)>1/RATE:raise ValueError('Metadata extension requires exact proven original silent-gap subrange')
     unit[key]=target
   for j,u in enumerate(units,1):u['sentence']=j;u['duration']=u['speech_end']-u['speech_start'];u['gap_from_previous']=None if j==1 else u['audible_start']-units[j-2]['audible_end']
   row.update(text=p['display_text'],speech_text=' '.join(u['speech_text'] for u in units),phonemes=' '.join(u.get('phonemes','') for u in units),sentences=units)
  else:
   if not ops:
    result.append(deepcopy(scene));continue
   row['duration']=scene['duration']
   offset=row['speech_start']-scene['speech_start']
   for u in row['sentences']:
    for key in ('speech_start','speech_end','audible_start','audible_end'):u[key]+=offset
  result.append(row)
 return result


def inputs_for(plan,unit,old,language,voice,runtime,renderer):
 code='b' if language=='en' and voice.startswith('b') else renderer.LANGUAGES[language][0]
 return dict(schema=1,speech_text=unit['speech_text'],speed=plan['effective_speed'],language=language,lang_code=code,dialect=renderer.narration_dialect(language,code,voice),voice=voice,sample_rate=RATE,synthesis_runtime=runtime)

def plans_for(renderer,lesson,old,language,voice):
 code='b' if language=='en' and voice.startswith('b') else renderer.LANGUAGES[language][0];dialect=renderer.narration_dialect(language,code,voice)
 if old.get('sample_rate')!=RATE or old.get('language')!=language or old.get('voice')!=voice or old.get('dialect')!=dialect:raise ValueError('Original voice/rate/dialect differs')
 plans=renderer.prepare_scene_plans(lesson,language,dialect,old['speed'],voice=voice)
 return plans,plan_track(plans,old,language)

def metadata(lesson,old,scenes,pcm,audio):
 return dict(schema=1,language=old['language'],voice=old['voice'],lang_code=old['lang_code'],dialect=old['dialect'],speed=old['speed'],sample_rate=RATE,total_duration=len(pcm)/RATE,media_sha256=digest(audio),media_bytes=audio.stat().st_size,render_inputs=dict(kind=KIND,lesson=lesson['id'],language=old['language'],voice=old['voice'],scenes=[dict(narration=s['narration']) for s in lesson['scenes']]),scenes=scenes,mixed_published_scene_audio=True)

def fresh_output(stage,path):
 stage,path=Path(stage).resolve(),Path(path)
 try:relative=path.relative_to(stage)
 except ValueError:raise ValueError('Output must belong to the independent stage')
 if '..' in relative.parts or not path.resolve().is_relative_to(stage):raise ValueError('Output leaves the stage')
 cursor=stage
 for part in relative.parts:
  cursor=cursor/part
  if cursor.is_symlink():raise ValueError('No output symlink path')
 if os.path.lexists(path):raise FileExistsError('Preserve earlier output')


def _copy(source,target):
 import shutil
 target.parent.mkdir(parents=True,exist_ok=True)
 with Path(source).open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b)

def render(stage,baseline,lessons):
 """Caller owns normal capped CPU/activity; all source/plans precede decoding."""
 lessons=render_selection(lessons)
 source,catalogs=sources(stage,baseline,lessons)
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('HF_HUB_OFFLINE')!='1' or Path(os.environ['SPACR_TUTORIAL_MODEL_CACHE']).resolve()!=common.MODELS or Path(os.environ['SPACR_TUTORIAL_SENTENCE_CACHE']).resolve()!=common.CACHE:raise ValueError('Normal offline CPU/shared model-cache runtime required')
 prepared=[]
 for lesson in lessons:
  root=Path(stage)/'production'/lesson
  if any(os.path.lexists(root/p) for p in (RECORD,WORK)):raise FileExistsError('Preserve earlier audio attempts')
  for track in source[lesson]['tracks']:
   language,voice=track['language'],track['voice'];output=root/'audio'/language/(voice+'.m4a')
   fresh_output(stage,output);fresh_output(stage,output.with_suffix('.json'))
   old=read(Path(source[lesson]['baseline_root'])/track['timing']);current=selected(catalogs['lessons_'+language+'.json'],lesson);plans,scope=plans_for(renderer,current,old,language,voice);prepared.append((lesson,track,current,old,plans,scope))
 # No model/cache/decode/output work can precede the entire batch's source/unit guard.
 import numpy as np
 from sentence_audio_cache import cached_sentence
 from verify_audio_release import _decode_track,_probe_track,DECODED_DURATION_TOLERANCE_SECONDS
 from narration_audio import ffmpeg_runtime_identity
 results={lesson:[] for lesson in lessons};model=None;pipelines={}
 for lesson in lessons:(Path(stage)/'production'/lesson/WORK).mkdir()
 for lesson,track,current,old,plans,scope in prepared:
  language,voice=track['language'],track['voice'];root=Path(stage)/'production'/lesson;folder=root/WORK/language/voice;folder.mkdir(parents=True);base=Path(source[lesson]['baseline_root']);audio=root/'audio'/language/(voice+'.m4a');audio.parent.mkdir(parents=True,exist_ok=True)
  if not scope['operations']:
   scenes=current_scenes(plans,old,scope,[],[]);gaps=declared_sound_gaps(scenes);_copy(base/track['audio'],audio)
   if scenes==old['scenes']:_copy(base/track['timing'],audio.with_suffix('.json'))
   else:write(audio.with_suffix('.json'),metadata(current,old,scenes,range(scope['declared_samples']),audio))
   results[lesson].append(dict(**track,mode='published_audio_copy',scope=scope,declared_sound_gaps=gaps,output_audio_sha256=digest(audio),output_timing_sha256=digest(audio.with_suffix('.json'))));continue
  _,_,errors=_probe_track(base/track['audio'])
  if errors:raise ValueError('Published source format failed')
  frames,duration,data,errors=_decode_track(base/track['audio'])
  if errors or abs(duration-old['total_duration'])>DECODED_DURATION_TOLERANCE_SECONDS:raise ValueError('Published source decode/duration failed')
  original=np.frombuffer(data,dtype='<i2');extensions=prove_extensions(original,old,scope,np);(folder/'source.s16le').write_bytes(data)
  runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu') if any(op['new_units'][0]<op['new_units'][1] for op in scope['operations']) else None;operations=[];replacements=[];timed=[]
  for number,op in enumerate(scope['operations']):
   unitfolder=folder/f'span-{number:03}';unitfolder.mkdir();plan=plans[op['scene']-1];units=plan['sentences'][slice(*op['new_units'])];raws=[];phones=[];input_rows=[]
   for unit in units:
    inputs=inputs_for(plan,unit,old,language,voice,runtime,renderer)
    def synthesize():
     nonlocal model
     import torch
     from kokoro import KPipeline,KModel
     if model is None:model=KModel(repo_id='hexgrad/Kokoro-82M',model=str(renderer.SNAPSHOT/'kokoro-v1_0.pth'),config=str(renderer.SNAPSHOT/'config.json')).to('cpu').eval()
     code=inputs['lang_code']
     if code not in pipelines:pipelines[code]=KPipeline(lang_code=code,repo_id='hexgrad/Kokoro-82M',model=model,device='cpu')
     pipeline=pipelines[code];loaded=pipeline.load_voice(str(renderer.SNAPSHOT/'voices'/str(voice+'.pt')))
     if hashlib.sha256(loaded.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=runtime['voice_pack']['tensor_sha256']:raise ValueError('Loaded synthesis voice differs')
     chunks=[];phonemes=[]
     for result in pipeline(inputs['speech_text'],voice=loaded,speed=inputs['speed']):
      if result.audio is not None:chunks.append(result.audio.detach().cpu().numpy().astype(np.float32))
      if result.phonemes:phonemes.append(str(result.phonemes))
     if not chunks:raise ValueError('Empty changed synthesis')
     return np.concatenate(chunks),' '.join(phonemes)
    raw,phonemes,reused=cached_sentence(common.CACHE,inputs,synthesize,np);raws.append(raw);phones.append(phonemes);input_rows.append(dict(inputs=inputs,phonemes=phonemes,cache_reused=reused,raw_sha256=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest()))
   assembled,unitrows=replacement_block(renderer,plan,units,raws,phones,op,np)
   if len(assembled):replacement,mastering=master_changed(assembled,unitfolder,renderer,lesson,language,voice,np)
   else:replacement=assembled;mastering=None
   (unitfolder/'replacement.s16le').write_bytes(replacement.tobytes());replacements.append(replacement);timed.append(unitrows);operations.append(dict(plan=op,units=input_rows,mastering=mastering,replacement_sha256=digest(unitfolder/'replacement.s16le')))
  joined,ranges=splice(original,scope['operations'],replacements,scope['declared_samples'],np);scenes=current_scenes(plans,old,scope,replacements,timed,extensions);gaps=declared_sound_gaps(scenes);cores=retained_proofs(original,joined,old,scope,replacements,np);(folder/'joined.s16le').write_bytes(joined.tobytes())
  subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','s16le','-ar',str(RATE),'-ac','1','-i',str(folder/'joined.s16le'),'-c:a','aac','-b:a','48k','-movflags','+faststart',str(audio)],check=True)
  peak=renderer.true_peak_dbfs(audio)
  if peak is None or peak>renderer.MAX_RELEASE_TRUE_PEAK_DBFS:raise ValueError('Codec peak failed; never globally re-master original PCM')
  write(audio.with_suffix('.json'),metadata(current,old,scenes,joined,audio))
  results[lesson].append(dict(**track,mode='source_coordinate_splice',scope=scope,runtime=runtime,operations=operations,decoder=ffmpeg_runtime_identity(),source_pcm_sha256=digest(folder/'source.s16le'),metadata_extensions=extensions,retained_units=cores,declared_sound_gaps=gaps,decoded_samples=frames,decoded_duration=duration,ranges=ranges,joined_pcm_sha256=digest(folder/'joined.s16le'),decoded_true_peak_dbfs=peak,output_audio_sha256=digest(audio),output_timing_sha256=digest(audio.with_suffix('.json'))))
 if sources(stage,baseline,lessons)!=(source,catalogs):raise ValueError('Frozen source changed during audio refresh')
 for lesson in lessons:
  root=Path(stage)/'production'/lesson;write(root/RECORD,dict(schema=1,kind=KIND,source=source[lesson],producer_sha256=digest(__file__),common_helper_sha256=digest(common.__file__),renderer_sha256=digest(renderer.__file__),tracks=results[lesson]))
 return [Path(stage)/'production'/lesson/RECORD for lesson in lessons]


def checked_tracks(stage,baseline,lesson,*,decode=False):
 """No original AAC decode/TTS/mastering. Explicit finalAAC acceptance opt-in."""
 if lesson not in POLICIES:raise ValueError('Lesson is not allowlisted')
 root=Path(stage)/'production'/lesson;record=read(common.regular(root,RECORD));source,catalogs=sources(stage,baseline,record['source']['selected_lessons']);expected_source=source[lesson]
 sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'));import render_all_voices as renderer
 import numpy as np
 from sentence_audio_cache import cached_sentence
 from narration_audio import ffmpeg_runtime_identity
 if record['schema']!=1 or record['kind']!=KIND or record['source']!=expected_source or record['producer_sha256']!=digest(__file__) or record['common_helper_sha256']!=digest(common.__file__) or record['renderer_sha256']!=digest(renderer.__file__):raise ValueError('Explicit mixed producer/source changed')
 inventory={(t['language'],t['voice']) for t in expected_source['tracks']}
 if len(record['tracks'])!=len(inventory) or {(t['language'],t['voice']) for t in record['tracks']}!=inventory:raise ValueError('Exact original offering inventory required')
 exact={f'{lang}/{voice}{suffix}' for lang,voice in inventory for suffix in ('.m4a','.json')}
 paths={p.relative_to(root/'audio').as_posix() for p in (root/'audio').rglob('*') if p.is_file()}
 if paths!=exact:raise ValueError('Exact normal audio output inventory required')
 checked=[]
 for original in expected_source['tracks']:
  language,voice=original['language'],original['voice'];row=next(r for r in record['tracks'] if (r['language'],r['voice'])==(language,voice));base=Path(expected_source['baseline_root']);old=read(base/original['timing']);current=selected(catalogs['lessons_'+language+'.json'],lesson);plans,scope=plans_for(renderer,current,old,language,voice);folder=root/WORK/language/voice;audio=common.regular(root,f'audio/{language}/{voice}.m4a');timing=common.regular(root,f'audio/{language}/{voice}.json')
  if any(row.get(k)!=v for k,v in original.items()) or row['scope']!=scope:raise ValueError('Original/unit-scope binding differs')
  if not scope['operations']:
   scenes=current_scenes(plans,old,scope,[],[]);expected=old if scenes==old['scenes'] else metadata(current,old,scenes,range(scope['declared_samples']),audio)
   if row['declared_sound_gaps']!=declared_sound_gaps(scenes) or row['mode']!='published_audio_copy' or digest(audio)!=original['audio_sha256'] or read(timing)!=expected:raise ValueError('Unchanged original AAC/timing projection differs')
  else:
   src=common.regular(folder,'source.s16le');original_pcm=np.frombuffer(src.read_bytes(),dtype='<i2');runtime=renderer.synthesis_runtime_identity(renderer.SNAPSHOT,voice,'cpu') if any(op['new_units'][0]<op['new_units'][1] for op in scope['operations']) else None
   if row['mode']!='source_coordinate_splice' or row['runtime']!=runtime or digest(src)!=row['source_pcm_sha256'] or len(original_pcm)!=row['decoded_samples'] or abs(row['decoded_duration']-len(original_pcm)/RATE)>1e-9 or row['decoder']!=ffmpeg_runtime_identity() or len(row['operations'])!=len(scope['operations']):raise ValueError('Original decoded PCM/runtime proof differs')
   extensions=prove_extensions(original_pcm,old,scope,np)
   if extensions!=row['metadata_extensions']:raise ValueError('Proven original metadata-gap extension differs')
   replacements=[];timed=[]
   for number,(op,saved) in enumerate(zip(scope['operations'],row['operations'])):
    plan=plans[op['scene']-1];units=plan['sentences'][slice(*op['new_units'])];unitfolder=folder/f'span-{number:03}';raws=[];phones=[]
    if saved['plan']!=op or len(saved['units'])!=len(units):raise ValueError('Replacement source/unit inventory differs')
    for unit,item in zip(units,saved['units']):
     inputs=inputs_for(plan,unit,old,language,voice,runtime,renderer)
     def unavailable():raise ValueError('Changed-unit cache missing; checker never synthesizes')
     raw,phonemes,_=cached_sentence(common.CACHE,inputs,unavailable,np)
     if item['inputs']!=inputs or item['phonemes']!=phonemes or item['raw_sha256']!=hashlib.sha256(raw.astype('<f4',copy=False).tobytes()).hexdigest():raise ValueError('Exact changed-unit cache differs')
     raws.append(raw);phones.append(phonemes)
    assembled,rows=replacement_block(renderer,plan,units,raws,phones,op,np);replacement_path=common.regular(unitfolder,'replacement.s16le');replacement=np.frombuffer(replacement_path.read_bytes(),dtype='<i2')
    if len(assembled):
     a,b=common.regular(unitfolder,'unmastered.s16le'),common.regular(unitfolder,'mastered.s16le');config=renderer.mastering_config(lesson,language,voice);expected=dict(configuration=config,filter=config['filters'][0],tool=ffmpeg_runtime_identity(),unmastered_sha256=digest(a),mastered_sha256=digest(b),samples=len(assembled))
     if expected['filter']!='loudnorm=I=-16:TP=-3:LRA=11' or a.read_bytes()!=assembled.tobytes() or b.stat().st_size!=len(assembled)*2 or b.read_bytes()!=replacement_path.read_bytes() or saved['mastering']!=expected:raise ValueError('Changed-only mastering differs')
    elif len(replacement) or saved['mastering'] is not None:raise ValueError('Deletion must contain no synthesized/mastered PCM')
    if digest(replacement_path)!=saved['replacement_sha256']:raise ValueError('Replacement PCM differs')
    replacements.append(replacement);timed.append(rows)
   joined,ranges=splice(original_pcm,scope['operations'],replacements,scope['declared_samples'],np);scenes=current_scenes(plans,old,scope,replacements,timed,extensions);gaps=declared_sound_gaps(scenes);cores=retained_proofs(original_pcm,joined,old,scope,replacements,np);p=common.regular(folder,'joined.s16le');peak=row['decoded_true_peak_dbfs']
   if type(peak) not in (int,float) or not math.isfinite(peak) or peak>renderer.MAX_RELEASE_TRUE_PEAK_DBFS or p.read_bytes()!=joined.tobytes() or digest(p)!=row['joined_pcm_sha256'] or ranges!=row['ranges'] or cores!=row['retained_units'] or gaps!=row['declared_sound_gaps'] or read(timing)!=metadata(current,old,scenes,joined,audio):raise ValueError('Exact complement/join/current timing differs')
  if digest(audio)!=row['output_audio_sha256'] or digest(timing)!=row['output_timing_sha256']:raise ValueError('Finished audio/timing bytes differ')
  if decode:
   from verify_audio_release import check_track
   errors=check_track(audio)
   if errors:raise ValueError('Normal final AAC acceptance failed: '+str(errors))
  checked.append(dict(language=language,voice=voice,audio_sha256=digest(audio),timing_sha256=digest(timing),provenance_kind=KIND,mixed_record_sha256=digest(root/RECORD),original_audio_sha256=original['audio_sha256']))
 return deepcopy(expected_source['voices']),checked


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);p.add_argument('--baseline',required=True,type=Path);p.add_argument('--lessons',nargs='+',choices=list(POLICIES),required=True);p.add_argument('--render',action='store_true',required=True);a=p.parse_args();print(*render(a.stage,a.baseline,render_selection(a.lessons)),sep='\n')
if __name__=='__main__':main()
