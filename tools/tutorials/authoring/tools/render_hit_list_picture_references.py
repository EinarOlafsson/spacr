"""One authored route card and a fixed narrow header; no GUI or science execution."""
from pathlib import Path
import argparse,json,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hit_list_picture_refresh as picture
LESSON=picture.LESSON
HEADER=('Hit List — standalone results browser','Browse saved results, filter candidates and export a shortlist.')
ROUTE=('Open a saved Regression result',[
 'From Home, open Regression and choose Hits.',
 'If the panel stays hidden, use the supplied companion:',
 'python hit_list_companion.py',
 'Browse to the extracted results folder.',
 'The standalone browser does not connect Investigate.'])
def regular(root,relative):return picture.regular(root,relative)
def checked_frames(folder):
 from PIL import Image
 for index,h in enumerate(picture.FRAME_HASHES,1):
  path=regular(folder,f'chapter-{index:03}.png')
  if picture.digest(path)!=h:raise ValueError('Reviewed source PNG changed')
  with Image.open(path) as image:
   if image.size!=(2560,1440) or image.format!='PNG':raise ValueError('Only reviewed2560x1440 source geometry')
   image.verify()
def definitions():return dict(header=list(HEADER),route=ROUTE,header_rectangle=list(picture.RECT),frame_edges=list(picture.EDGES),source_frame_hashes=list(picture.FRAME_HASHES))
def contract(current,source,repo):
 return dict(schema=1,kind=picture.KIND,lesson=LESSON,english_sha256=picture.identity(current),source_sha256=picture.identity(source),definitions_sha256=picture.identity(definitions()),generator_sha256=picture.digest(__file__),style_sha256=picture.digest(Path(repo)/'tools/tutorials/authoring/tools/render_install_keyframes.py'))
def validate_references(folder,current,source,repo):
 from PIL import Image
 folder=Path(folder);picture.checked_english(current);checked_frames(folder/'source-frames')
 value=json.loads(regular(folder,'references.json').read_text())
 if {k:v for k,v in value.items() if k!='images'}!=contract(current,source,repo):raise ValueError('Route/header source or wording binding differs')
 images=value.get('images',{})
 if set(images)!= {'route','header'}:raise ValueError('Exactly one route and one header image')
 for key,size,name in [('route',(3840,2160),'scene-001--route.png'),('header',(3840,72),'header.png')]:
  p=regular(folder,name);expected=dict(image=name,sha256=picture.digest(p),size=list(size),kind='authored_route_reference' if key=='route' else 'authored_feature_header')
  if images[key]!=expected:raise ValueError('Reference geometry/image differs')
  with Image.open(p) as image:
   if image.size!=size or image.format!='PNG' or image.mode!='RGB':raise ValueError('Opaque RGB reference geometry required')
   image.verify()
 return images

def render(stage,frames,destination,repo):
 from PIL import Image,ImageDraw
 stage,repo,destination=Path(stage),Path(repo),Path(destination)
 source,current,clock=picture.checked_source(stage);checked_frames(frames)
 if destination.exists() or destination.is_symlink():raise FileExistsError('Preserve earlier references')
 if destination.resolve()!=(stage/'production'/LESSON/'picture-references').resolve():raise ValueError('References belong only to their explicit stage')
 sys.path.insert(0,str(repo/'tools/tutorials/authoring/tools'));import render_install_keyframes as style
 destination.mkdir(parents=True);saved=destination/'source-frames';saved.mkdir()
 for index in range(1,12):shutil.copy2(regular(frames,f'chapter-{index:03}.png'),saved/f'chapter-{index:03}.png')
 route=style.gradient_background();draw=ImageDraw.Draw(route)
 style.draw_text(draw,(150,100),'Hit List · route reference',style.SUBTITLE,style.CYAN)
 style.draw_text(draw,(150,190),ROUTE[0],style.TITLE,style.TEXT)
 for i,line in enumerate(ROUTE[1]):
  if style.BODY.getlength(line)>style.scaled(1620):raise ValueError('Route text exceeds safe width')
  style.draw_text(draw,(170,360+i*115),line,style.BODY,style.TEXT)
 style.draw_text(draw,(150,1000),'spaCR tutorial series',style.SMALL,style.MUTED)
 # Header is physically72px tall, not the normal1080 logical canvas. Opaque
 # pixels only inside approved ROI; use the same normal font at26physicalpx.
 header=Image.new('RGB',(picture.RECT[2],picture.RECT[3]),style.BG_TOP);draw=ImageDraw.Draw(header);font=style.font(13)
 for text,y in zip(HEADER,(5,37)):
  bounds=draw.textbbox((12,y),text,font=font)
  if bounds[0]<0 or bounds[2]>3840-12 or bounds[3]>72:raise ValueError('Header text leaves approved rectangle')
  draw.text((12,y),text,font=font,fill=style.TEXT)
 images={}
 for key,image,name in [('route',route,'scene-001--route.png'),('header',header,'header.png')]:
  image.save(destination/name);images[key]=dict(image=name,sha256=picture.digest(destination/name),size=list(image.size),kind='authored_route_reference' if key=='route' else 'authored_feature_header')
 (destination/'references.json').write_text(json.dumps({**contract(current,source,repo),'images':images},ensure_ascii=False,indent=2)+'\n')
 validate_references(destination,current,source,repo)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('stage','repo','source-frames','destination'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();render(a.stage,a.source_frames,a.destination,a.repo)
if __name__=='__main__':main()
