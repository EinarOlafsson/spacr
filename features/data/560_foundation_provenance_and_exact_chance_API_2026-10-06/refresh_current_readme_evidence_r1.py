from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path.cwd()
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
names = subprocess.check_output(['git', 'ls-files', '-z']).split(b'\0')
size = sum((root / os.fsdecode(name)).stat().st_size for name in names if name and (root / os.fsdecode(name)).is_file())
measured_mb = round(size / 1048576)
readme = root / 'README.rst'
text = readme.read_text()
assert '1029 MB checkout (measured 2026-10-05)' in text
readme.write_text(text.replace('1029 MB checkout (measured 2026-10-05)', f'{measured_mb} MB checkout (measured 2026-10-06)'))
source = 'The standard installation includes the Qt desktop interface. For a server, cluster or CI runner, run the command-line pipelines without opening it:'
targets = {
    'sv': 'Standardinstallationen innehåller Qt-gränssnittet för skrivbordet. På en server, ett kluster eller en CI-körmiljö kan du köra kommandoradspipelines utan att öppna det:',
    'de': 'Die Standardinstallation enthält die Qt-Desktopoberfläche. Auf einem Server, Cluster oder CI-Runner können Sie die Befehlszeilen-Pipelines ausführen, ohne sie zu öffnen:',
    'es': 'La instalación estándar incluye la interfaz de escritorio Qt. En un servidor, clúster o ejecutor de CI, ejecute los flujos de trabajo de línea de comandos sin abrirla:',
    'zh_CN': '标准安装包含 Qt 桌面界面。在服务器、集群或 CI 运行器上，可运行命令行流程而不打开该界面：',
    'pt': 'A instalação padrão inclui a interface de desktop Qt. Em um servidor, cluster ou executor de CI, execute os fluxos de trabalho de linha de comando sem abri-la:',
    'hi': 'मानक इंस्टॉलेशन में Qt डेस्कटॉप इंटरफ़ेस शामिल है। सर्वर, क्लस्टर या CI रनर पर इसे खोले बिना कमांड-लाइन पाइपलाइन चलाएँ:',
    'ko': '표준 설치에는 Qt 데스크톱 인터페이스가 포함됩니다. 서버, 클러스터 또는 CI 실행기에서는 인터페이스를 열지 않고 명령줄 파이프라인을 실행하세요:',
    'is': 'Hefðbundin uppsetning inniheldur Qt-skjáborðsviðmótið. Á þjóni, reikniklasa eða CI-keyrsluumhverfi má keyra skipanalínuverkflæðin án þess að opna það:',
    'fr': 'L’installation standard inclut l’interface de bureau Qt. Sur un serveur, un cluster ou un exécuteur CI, lancez les pipelines en ligne de commande sans l’ouvrir :',
}
for language, target in targets.items():
    folder = root / 'docs/i18n/reviewed/readme' / language
    old = json.loads((folder / '2026-10-05-checkout-size.json').read_text())
    record = old['records'][0]
    for field in ('source', 'translation'):
        assert '1029' in record[field] and '2026-10-05' in record[field]
        record[field] = record[field].replace('1029', str(measured_mb)).replace('2026-10-05', '2026-10-06')
    record['source_sha256'] = hashlib.sha256(record['source'].encode()).hexdigest()
    old['records'].append({'label': 'standard-Qt-installation-headless-command-line-use',
                           'source': source, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                           'translation': target})
    old['review_method'] = 'Direct Codex AI numeric review against all current tracked file sizes, retaining the original dated download measurements, and technical review of standard Qt installation versus headless execution. No native-speaker signoff.'
    path = folder / '2026-10-06-current-installation.json'
    assert not path.exists()
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')
receipt = {'tracked_bytes': size, 'README_MB_binary_units': measured_mb,
           'measured_date': '2026-10-06', 'tracked_file_count': sum(bool(name) for name in names),
           'old_measured_download_figures_retained': True,
           'removed_redundant_installer_and_module_navigation_paragraphs': True,
           'nine_source_bound_installation_reviews_written': True}
(scratch / 'current-readme-measurement-r1.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
