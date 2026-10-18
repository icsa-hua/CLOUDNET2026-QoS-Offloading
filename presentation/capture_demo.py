"""Capture actual Rich dashboard output as lossless terminal images."""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.segment import Segment
from demo import Demo

OUT = ROOT / 'presentation' / 'assets'
FONT = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 18)

def capture(demo, name):
    console = Console(file=io.StringIO(), width=156, height=48,
                      force_terminal=True, color_system='truecolor', record=True)
    console.print(demo.render(console))
    lines = list(Segment.split_lines(console._record_buffer))
    image = Image.new('RGB', (156 * 11 + 32, len(lines) * 24 + 32), '#0c1018')
    draw = ImageDraw.Draw(image)
    for y, line in enumerate(lines):
        x = 16
        for segment in line:
            if segment.control:
                continue
            style = segment.style
            color = '#eeeeee'
            if style and style.color:
                color = style.color.get_truecolor().hex
            length = FONT.getlength(segment.text)
            if style and style.bgcolor:
                draw.rectangle((x, 16+y*24, x+length, 40+y*24),
                               fill=style.bgcolor.get_truecolor().hex)
            draw.text((x, 16+y*24), segment.text, font=FONT, fill=color)
            x += length
    image.save(OUT / f'{name}.png')
    (OUT / f'{name}.txt').write_text(console.export_text())
    print(name, 'lines:', len(lines), 'size:', image.size)
    for i,line in enumerate((OUT / f'{name}.txt').read_text().splitlines()):
        print(f'{i:02d} {line[:110]}')

demo = Demo(outage_at=0)
for _ in range(19):
    demo.step()
capture(demo, 'dashboard-normal')
demo.inject('mec_outage')
demo.step()
capture(demo, 'dashboard-cloud')
demo.inject('blackout')
for _ in range(8):
    demo.step()
capture(demo, 'dashboard-blackout')
