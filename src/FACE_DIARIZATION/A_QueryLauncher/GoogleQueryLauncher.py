import os
import time
import requests
import base64
import io
from PIL import Image
import hashlib
from selenium import webdriver
from src.FACE_DIARIZATION.A_QueryLauncher.QueryLauncher import QueryLauncher
from src.BaseArgs import QueryArgs
import pandas as pd
from selenium.common import exceptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import urllib

class GoogleQueryLauncher(QueryLauncher):
    def __init__(self,participants_path, chrome_driver_path, logs_path, output_dir,
                 imgs2download=150,extra_info=None):
        super().__init__(participants_path, chrome_driver_path, logs_path, output_dir,
                         imgs2download=imgs2download, extra_info=extra_info)

    def persist_image(self,folder_path:str,url:str, index_img = '0'):
        success = True
        if 'data:image/jpeg;base64' in url:
            try:
                image_content = url[url.find('/9'):]
                image_content = base64.b64decode(image_content)
                file_path = os.path.join(folder_path, index_img+"_"+hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
                image = io.BytesIO(image_content)

                with open(file_path, 'wb') as f:
                    #im = Image.open(image).save(f, "JPEG", quality=85)
                    im = Image.open(image).save(f, "JPEG", quality=100)
                print(f"1.SUCCESS - saved {url} - as {file_path}")
            except Exception as e:
                print(f"1.ERROR - Could not save {url} - {e}")
                success = False
            finally:
                return success
        else:
            try:
                if("55" in index_img):
                    print("debug")
                image_content = requests.get(url, timeout=7).content
                #urllib.request.urlretrieve("http://www.gunnerkrigg.com//comics/00000001.jpg", "00000001.jpg")
            except Exception as e:
                print(f"2.ERROR - Could not download {url} - {e}")
                success = False

            try:
                image_file = io.BytesIO(image_content)
                image = Image.open(image_file).convert('RGB')
                file_path = os.path.join(folder_path,index_img+"_"+hashlib.sha1(image_content).hexdigest()[:10] + '.png')
                image.save(file_path, "PNG", quality=100)
                #f = Image.open(file_path)
                #f.close()
                # with open(file_path, 'wb') as f:
                #     #image.save(f, "JPEG", quality=85)
                #     image.save(f, "JPEG", quality=100)
                print(f"2.SUCCESS - saved {url} - as {file_path}")
                success =True
            except Exception as e:
                print(f"3.ERROR - Could not save {url} - {e}")
                success = False

            finally:
                return success
        return success



    def fetch_image_urls(self,query:str, wd:webdriver, sleep_between_interactions:int=1, max_timeout = 5, imgs_offset = 5):
        def scroll_to_end(wd):
            wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(sleep_between_interactions)

        # build the google query
        search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

        # load the page
        wd.get(search_url.format(q=query))

        image_urls = list()
        reference_imgs_urls = list()
        wait = WebDriverWait(wd, max_timeout)
        results_start = 0
        #wd.window_handles
        while len(image_urls) < (self.imgs2download+imgs_offset):
            scroll_to_end(wd)
            # get all image thumbnail results
            #thumbnail_results = wd.find_elements_by_css_selector("img.rg_ic")
            wait.until(EC.presence_of_element_located((By.ID, "islrg")))
            thumbnail_div = wd.find_element_by_id('islrg')
            WebDriverWait(thumbnail_div, max_timeout).until(EC.presence_of_element_located((By.CLASS_NAME, "islrc")))
            thumbnail_div = thumbnail_div.find_elements_by_class_name('islrc')[0]
            WebDriverWait(thumbnail_div, max_timeout).until(EC.presence_of_element_located((By.TAG_NAME, "div")))
            div_with_link_img = thumbnail_div.find_elements_by_tag_name('div')
            n_found_divs = len(div_with_link_img)
            for div_of_img in div_with_link_img[results_start:n_found_divs]:
                try:
                    if(len(image_urls)>=(self.imgs2download+imgs_offset)):
                        break
                    wd.switch_to_window(wd.window_handles[0])
                    possible_img_link = div_of_img.find_elements_by_tag_name('a')
                    #try:
                    for pos_link in possible_img_link:
                        possible_imgs = pos_link.find_elements_by_tag_name('img')
                        p_links = pos_link.get_attribute("href")
                        print("PL: ", p_links)
                        if(p_links!=None):
                            reference_imgs_urls.append(p_links)

                        #Click on imgs in order to let the link appear
                        for img in possible_imgs:
                            w, h = int(img.get_attribute("width")), int(img.get_attribute("width"))
                            if (w < 60 or h < 60):
                                possible_imgs.remove(img)
                                continue
                            else:
                                img.click()
                                #Wait until click have had effect
                                time.sleep(sleep_between_interactions)
                        if(len(possible_imgs)>0):
                            new_img_url = pos_link.get_attribute("href")
                            if(new_img_url != None):
                                #print("URL IMG:  ", new_img_url)
                                wd.execute_script("window.open()")
                                wd.switch_to_window(wd.window_handles[1])
                                wd.get(new_img_url)
                                #wait until load new page
                                time.sleep(sleep_between_interactions)
                                wait.until(EC.presence_of_element_located((By.TAG_NAME, "img")))
                                big_imgs = wd.find_elements_by_tag_name('img')
                                for big_img_index in range(len(big_imgs)):
                                    w, h = int(big_imgs[big_img_index].get_attribute("width")), int(big_imgs[big_img_index].get_attribute("width"))
                                    if (w < 60 or h < 60):
                                        continue
                                    else:
                                        print("IMG:", big_imgs[big_img_index].get_attribute("src"))
                                        image_urls.append(big_imgs[big_img_index].get_attribute("src"))
                                        break
                                wd.close()
                                wd.switch_to_window(wd.window_handles[0])
                except Exception as e:
                    print(f"ERROR - {e} (continue ...)")
                    for i in range(1,len(wd.window_handles)):
                        wd.close()
                    wd.switch_to_window(wd.window_handles[0])

            #LOAD MORE
            #Press load button if not enough imgs
            if len(image_urls) >= (self.imgs2download+imgs_offset):
                print(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")
                # move the result startpoint further down
                results_start = n_found_divs

        return image_urls, reference_imgs_urls

    def save_URLs(self,urls_list, extra_name="IMG_LINKS_"):
        os.makedirs(self.logs_path, exist_ok=True)
        path_urls = os.path.join(self.logs_path, extra_name+self.current_participant.replace(" ", "_")+".csv")
        pd.DataFrame(list(urls_list),columns=["Url_link"]).to_csv(path_urls, index=False, header=True, sep=";")

    def download_imgs_from_keywords(self, img_directory="", sleep_between_interactions=1):
        #target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
        participant_name = self.current_participant
        final_url_list = []
        final_reference_list = []
        target_folder = os.path.join(self.output_dir, participant_name.replace(" ", "_"), img_directory)
        if (os.path.isdir(target_folder) and
            len(os.listdir(target_folder))>10):
            print(participant_name + ' have their images downloaded yet')
        else:
            #download images
            os.makedirs(target_folder, exist_ok=True)

            with webdriver.Chrome(executable_path=self.chrome_driver_path) as wd:
                urls_list, reference_list = self.fetch_image_urls(participant_name, wd=wd, sleep_between_interactions=sleep_between_interactions)

            n = 0
            num_digits = str(len(str(len(urls_list)+1))+1)
            for elem in urls_list:
                index_img = str("{0:0="+num_digits+"d}").format(n)
                success = self.persist_image(target_folder,elem,index_img=index_img)
                if(success):
                    final_url_list.append(elem)
                    final_reference_list.append(reference_list[n])
                n+=1
            self.save_URLs(final_url_list)
            self.save_URLs(final_reference_list, extra_name="REFERENCE_URL_")
            wd.quit()
        return final_url_list


if __name__ == "__main__":
    #create QueryLauncher oject:
    google_launcher_args_obj = QueryArgs()
    args = google_launcher_args_obj.parse()
    google_launcher = GoogleQueryLauncher(args.participants_path, args.chrome_driver_path,
                                     args.logs_path, args.output_dir,args.imgs_2_download,args.extra_info)
    for participant in google_launcher.participants:
        #download imgs of participant
        google_launcher.set_current_participant(participant)
        google_launcher.download_imgs_from_keywords()
        #Wait until images are completely downloaded
        time.sleep(3)
        # remove corrupted imgs:
        google_launcher.clean_corrupted_imgs()
        # rename images to more precisse name & remove corrupted images
        google_launcher.rename_and_sort_imgs()

