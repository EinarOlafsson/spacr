import sys,json
from PIL import Image, ImageDraw, ImageFont
P='/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/production/'
S='/mnt/firecuda2/Claude/toxoplasma_projects/presentation/images/'
jobs=json.load(open(sys.argv[1]))
tiles=[]
for name,(frame,box) in jobs.items():
    im=Image.open(frame if frame.startswith('/') else P+frame.replace('/','/keyframes/')+'.png').convert('RGB')
    x,y,w,h=box; c=im.crop((x,y,x+w,y+h))
    if c.width>2400: c=c.resize((2400,round(c.height*2400/c.width)),Image.LANCZOS)
    c.save(S+'keyframes/'+name+'.png'); t=c.copy(); t.thumbnail((640,400)); tiles.append((name,t))
W=3;H=(len(tiles)+W-1)//W
sh=Image.new('RGB',(W*650,H*430),'white'); f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18)
for i,(n,t) in enumerate(tiles):
    sh.paste(t,((i%W)*650,(i//W)*430+25)); ImageDraw.Draw(sh).text(((i%W)*650,(i//W)*430+2),n+f' {t.size}',fill='red',font=f)
sh.save('/tmp/claude-1000/-mnt-firecuda2-Claude-repo/d76975b5-7a10-46fa-bf01-dd62a1467de7/scratchpad/deck2/cropcheck.jpg',quality=85)
