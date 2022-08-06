import os
from fpdf import FPDF
from PIL import Image

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def main():
    #read all files from a given directory
    path = "./figures"
    imagelist = os.listdir(path)
    #print(imagelist)

    #Creating first pdf page image
    image1 = Image.open(path + '\\' + imagelist[0])
    image2 = Image.open(path + '\\' + imagelist[1])
    get_concat_v(image1, image2).save('./pdf/concat1.png')

    #Creating second pdf page image
    image3 = Image.open(path + '\\' + imagelist[2])
    image4 = Image.open(path + '\\' + imagelist[3])
    image6 = Image.open(path + '\\' + imagelist[5])
    imageaux = get_concat_v(image3, image4)
    get_concat_v(imageaux, image6).save('./pdf/concat2.png')

    imagelist2 = ['concat1.png', 'concat2.png', './figures/z_network_graph.png']

    #Creating pdf
    pdf = FPDF()
    pdf.set_author('March')
    #First page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(90, 9, 'Cumulative Returns & Drawdowns', 1, 0, 'C')
    pdf.ln(10)
    pdf.image('pdf\\' + imagelist2[0], w=195, h=230)
    #Second page
    pdf.add_page()
    pdf.set_author('March')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 8, 'Portfolio Analysis', 1, 0, 'C')
    pdf.ln(10)
    pdf.image('pdf\\' + imagelist2[1], w=195, h=250)
    #Third page
    pdf.add_page()
    pdf.set_author('March')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 8, 'Hierarchical Network', 1, 0, 'C')
    pdf.ln(10)
    pdf.image(imagelist2[2], w=195, h=250)

    #Saving pdf
    pdf.output("./pdf/portfolio_analysis.pdf", "F")

if __name__=='__main__':
    main()


