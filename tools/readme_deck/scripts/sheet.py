from PIL import Image, ImageDraw, ImageFont
import os
names=open('/tmp/claude-1000/-mnt-firecuda2-Claude-repo/d76975b5-7a10-46fa-bf01-dd62a1467de7/scratchpad/deck2/cands.txt').read().split()
names=[n for n in names if os.path.exists(n.split('/')[0]+'/keyframes/'+n.split('/')[1]+'.png')]
missing=[n for n in open('/tmp/claude-1000/-mnt-firecuda2-Claude-repo/d76975b5-7a10-46fa-bf01-dd62a1467de7/scratchpad/deck2/cands.txt').read().split() if n not in names]; print('missing',missing)
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
per=9
for k in range(0,len(names),per):
    sub=names[k:k+per]
    sh=Image.new('RGB',(3*960,3*570),'white')
    for i,n in enumerate(sub):
        d,b=n.split('/')
        im=Image.open(f'{d}/keyframes/{b}.png').convert('RGB'); im.thumbnail((960,540))
        c=Image.new('RGB',(960,570),'white'); c.paste(im,(0,30)); ImageDraw.Draw(c).text((4,2),n,fill='red',font=f)
        sh.paste(c,((i%3)*960,(i//3)*570))
    sh.save(f'/tmp/claude-1000/-mnt-firecuda2-Claude-repo/d76975b5-7a10-46fa-bf01-dd62a1467de7/scratchpad/deck2/kf{k//per:02d}.jpg',quality=85)
print(len(names))
