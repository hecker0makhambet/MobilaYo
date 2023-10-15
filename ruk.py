import flet as ft
import numpy as np
import json
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array
import os


classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_',
           'Corn_(maize)___healthy',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)',
           'Grape___healthy',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach___healthy',
           'Pepper,_bell___Bacterial_spot',
           'Pepper,_bell___healthy',
           'Potato___Early_blight',
           'Potato___healthy',
           'Potato___Late_blight',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___healthy',
           'Strawberry___Leaf_scorch',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___healthy',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


def load_model(json_file='model.json', h5_file="model.h5"):
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_file)
    # print("Loaded model from disk")
    return loaded_model


def make_prediction(loaded_model, formatted_image):
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                         metrics=['accuracy'])
    score = loaded_model.predict(formatted_image)
    return score


def format_image(image_name):
    test_image = Image.open(image_name)
    test_image = test_image.resize((224, 224), Image.LANCZOS)
    test_image = img_to_array(test_image) / 255
    image_batch = np.expand_dims(test_image, axis=0)
    return image_batch


disease_description = {
    'Potato___Early_blight': """Ранний фитофтороз - это в первую очередь заболевание стрессовых или стареющих растений. Симптомы появляются первыми на самой старой листве. На пораженных листьях образуются округлые или угловатые темно-коричневые повреждения диаметром от 0,12 до 0,16 дюйма (3-4 мм). В очагах поражения часто образуются концентрические кольца, создающие характерный эффект доски-мишени. Сильно пораженные листья желтеют и опадают. На зараженных клубнях появляется коричневая, покрытая коркой сухая гниль.

Комментарии к болезни
В промежутках между посевами грибок ранней фитофтороза может зимовать на картофельных отходах в поле, в почве, на клубнях и на других пасленовых растениях. Заражение происходит, когда споры гриба вступают в контакт с восприимчивыми листьями и присутствует достаточное количество свободной влаги. Прорастанию спор и заражению благоприятствует теплая погода и влажные условия от росы, дождя или дождевального орошения. Попеременно влажные и засушливые периоды с относительно сухими, ветреными условиями благоприятствуют рассеиванию спор и распространению болезней. Клубни могут быть заражены, когда их поднимают из почвы во время сбора урожая. При наличии достаточного количества влаги споры прорастают и заражают клубни.

Управление
Ранний фитофтороз можно свести к минимуму, поддерживая оптимальные условия выращивания, включая правильное внесение удобрений, орошение и борьбу с другими вредителями. Выращивайте сорта с более поздним сроком созревания и более длительным сроком хранения. Применение фунгицидов оправдано только в том случае, если заболевание запущено достаточно рано, чтобы привести к экономическим потерям. Следите за симптомами заболевания во время рутинного мониторинга и ведите учет своих результатов. При необходимости применяйте фунгициды сразу же при появлении симптомов; для дальнейшей защиты требуется применение с интервалом в 7-10 дней.
""",
    'Corn_(maize)___Common_rust_': """Обыкновенная кукурузная ржавчина, вызываемая грибком Puccinia sorghi, является наиболее часто встречающимся из двух основных ржавчинных заболеваний кукурузы в США, но она редко приводит к значительным потерям урожая кукурузы на полях штата Огайо (dent). Иногда полевая кукуруза, особенно в южной половине штата, действительно серьезно поражается, когда погодные условия благоприятствуют развитию и распространению ржавчинного гриба. Сладкая кукуруза, как правило, более восприимчива, чем полевая кукуруза. В годы с исключительно прохладным летом, и особенно на полях с поздним севом сахарной кукурузы, могут произойти потери урожая, когда листья в початках и над ними сильно поражаются до завершения налива зерна.
Симптомы
Хотя на кукурузных полях в течение всего вегетационного периода всегда можно обнаружить несколько пустул ржавчины, симптомы, как правило, проявляются только после удаления кисточек. Их можно легко распознать и отличить от других заболеваний по развитию темных, красновато-коричневых пустул (урединий), разбросанных как по верхней, так и по нижней поверхности листьев кукурузы (рис. 1). Эти пустулы могут появиться на любой надземной части растения, но наиболее обильны на листьях. Пустулы имеют овальную или удлиненную форму, как правило, небольшие, менее 1/4 дюйма в длину, и окружены эпидермальным слоем листа, где он прорвался. Если инфекция возникает, когда листья еще находятся в завитке, эти пустулы могут образовываться полосами по всей поверхности по мере увеличения листа в размерах
Цикл заболеваний и эпидемиология
В отличие от большинства других внекорневых болезней кукурузы, этот ржавчинный гриб не зимует в растительных остатках. Споры должны быть занесены ветром на север в течение вегетационного периода (с середины июня по середину июля) из субтропических и тропических регионов, где этот гриб выживает на кукурузе или древесном щавеле, альтернативных хозяевах. Молодые листья, как правило, более восприимчивы к инфекции, чем более старые. Развитию и распространению ржавчины способствуют длительные периоды низких температур в диапазоне от 60° до 74°F и высокая относительная влажность. В этих условиях пустулы развиваются на восприимчивых гибридах кукурузы и сортах сладкой кукурузы в течение 7 дней после заражения.

Иногда при тяжелых инфекциях возникает хлороз и отмирание листьев и листовых влагалищ. Уредоспоры, образующиеся в течение сезона, распространяются ветром, распространяя возбудителя на новые листья, растения и поля. По мере созревания растения кукурузы пустулы приобретают коричневато-черный цвет из-за развития более темных пигментированных телий, которые заменяют урединии и продуцируют телиоспоры. В тропических регионах телиоспоры заражают альтернативного хозяина, древесный щавель (вид Oxalis), однако в районах с умеренным климатом, таких как Огайо и другие штаты США. кукурузная лента, гриб не поражает древесный щавель, а телиоспоры не имеют реального эпидемиологического значения (не участвуют в цикле заболеваний).

Управление
Хотя ржавчина часто встречается на кукурузе в Огайо, очень редко возникала необходимость в применении фунгицидов. Это связано с тем, что существуют высокоустойчивые гибриды полевой кукурузы, и большинство из них обладают определенной степенью устойчивости. Однако попкорн и сладкая кукуруза могут быть весьма восприимчивыми. В сезоны, когда на нижних листьях перед шелушением присутствует значительная ржавчина, а погода не по сезону прохладная и влажная, для эффективной борьбы с болезнями может потребоваться раннее применение фунгицидов. Для борьбы с ржавчиной существует множество фунгицидов. Ознакомьтесь с последними рекомендациями по повышению эффективности в вашем местном окружном бюро по распространению информации и на веб-сайте C.O.R.N. Всегда читайте этикетку с фунгицидами, чтобы узнать нормы и сроки применения.
""",
    'Tomato___Early_blight': """Ранний фитофтороз -Alternaria linariae (=A. tomatophila; ранее известный как A. solani)

Инфекции начинаются с появления небольших коричневых пятен на старых листьях, которые быстро увеличиваются. Очаги поражения обычно окружает желтый ореол. На поражениях образуется "бычий глаз" в виде концентрических колец, которые можно увидеть с помощью ручной линзы. Отдельные очаги поражения увеличиваются и срастаются и могут привести к гибели целых листьев. Болезнь также может распространяться на стебли и плоды и вызывать темные поражения.
Это очень распространенное внекорневое заболевание растений томатов, которое может привести к дефолиации и снижению урожайности. Он также может поражать баклажаны. Гриб зимует в почве и на растительных остатках.  Он также может передаваться через семена и трансплантаты. Ранний фитофтороз обычно проявляется в результате выпадения осадков на нижние листья в начале сезона.
Когда листья отмирают, плоды становятся более уязвимыми к солнечным ожогам. Зараженные, отмершие листья могут прилипать к плодам. Болезнь может распространяться во влажную или сухую погоду, но ей благоприятствуют осадки и обильная роса. Споры болезни разносятся ветром, что позволяет болезни распространяться по саду или окрестностям.
Управление
Обеспечьте достаточное расстояние для улучшения циркуляции воздуха и удалите все присоски, которые выходят из основания растения
Тщательно следите за трансплантатами на предмет наличия признаков этого заболевания.
Хорошо мульчируйте растения, чтобы свести к минимуму разбрызгивание почвы.
Поливайте растения вокруг их основания. Избегайте намокания листвы.
Обрежьте нижние ветви с 3-4 листьями, как только растения хорошо укоренятся и начнут плодоносить.
Удаляйте зараженные листья в течение вегетационного периода и удаляйте все зараженные части растений в конце сезона.
Применяйте синтетический фунгицид или органический фунгицид (фиксированная медь) в соответствии с указаниями на этикетке в начале сезона, когда появляются симптомы, чтобы замедлить распространение болезни. Это может быть полезно в тех случаях, когда болезнь ежегодно вызывает серьезные поражения, приводящие к снижению урожайности.
Пораженные части растений можно измельчить и компостировать, если используются методы "горячего компостирования" (температура штабеля должна превышать 120 ° F по всей длине, а штабеля следует переворачивать два-три раза).
""",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': """Симптомы и признаки
Желтое скручивание листьев томата - это заболевание томатов, вызываемое вирусом желтого скручивания листьев томата. В марте 2007 года он был впервые обнаружен в Калифорнии и в настоящее время имеет ограниченное распространение. В то время была создана учебная брошюра (PDF), которая доступна для печати.

Зараженные растения томатов первоначально демонстрируют задержку роста и прямостоячие растения; растения, инфицированные на ранней стадии роста, будут сильно отставать в росте. Однако наиболее диагностическими симптомами являются те, что содержатся в листьях.

Листья зараженных растений маленькие и скручиваются кверху; на них заметно сильное сминание и пожелтение междоузлий и краев. Междоузлия зараженных растений укорачиваются, и вместе с задержкой роста растения часто приобретают кустистый вид, который иногда называют ростом, похожим на бонсай или брокколи. Цветки, образующиеся на зараженных растениях, обычно не развиваются и опадают (опадают). Производство плодов резко сокращается, особенно если растения заражены в раннем возрасте, и нередко на полях с сильно зараженными растениями наблюдаются потери в размере 100%.

Комментарии к болезни
Вирус желтого скручивания листьев томатов, несомненно, является одним из наиболее опасных патогенов томатов, и он ограничивает производство томатов во многих тропических и субтропических районах мира. Это также проблема во многих странах со средиземноморским климатом, таких как Калифорния. Таким образом, распространение вируса по всей Калифорнии следует рассматривать как серьезную потенциальную угрозу для томатной промышленности.

Существует ряд факторов, по которым она до сих пор не распространилась на все основные районы производства томатов в Калифорнии, включая долины Сакраменто и Сан-Хоакин. Во-первых, ее переносчик, вид белокрылки Bemisia, обычно не встречается в этих районах выращивания томатов, поскольку она там непереносима к зимним температурам. Во-вторых, зимний сезон в Центральной долине обеспечивает "естественный" период без помидоров, который обычно длится с конца ноября по начало февраля. Хотя вирус может поражать другие растения, томат является хозяином, в котором он накапливается быстрее всего. Таким образом, благодаря ежегодному "периоду без помидоров", вполне вероятно, что количество вирусного инокулята (а также популяций белокрылки) значительно сократится к тому времени, когда снова начнется сезон посадки томатов в конце зимы - начале весны. Это означало бы, что, даже если вирус сможет перезимовать, может потребоваться много времени, чтобы достичь уровней, наносящих экономический ущерб.

Вирус желтого скручивания листьев томатов относится к геминивирусу (семейство Geminiviridae). Несмотря на то, что вирус может поражать относительно широкий спектр видов растений, томат является хозяином, к которому вирус лучше всего адаптирован, и это способствует накоплению вируса и высокому уровню заболеваемости в полевых условиях. Другие хозяева включают пасленовые культуры, у которых могут развиться бессимптомные инфекции, и сорняки (например, паслен и дурман).

Кроме того, вирус вызывает скручивание листьев у некоторых сортов фасоли обыкновенной (Phaseolus vulgaris) и декоративного растения лизиантус (Eustoma grandiflorum). Целый ряд сорняков из других семейств могут быть заражены этим вирусом, но у большинства из них не развиваются явные симптомы заболевания. Неизвестно, насколько хорошо белокрылки заражаются вирусом от хозяев без симптомов. Однако была выдвинута гипотеза, что эти хозяева служат "мостиком" для вируса в отсутствие посевов томатов и что многолетние сорняки помогают вирусу закрепиться надолго.

Основным способом распространения вируса на короткие расстояния является белокрылка Bemisia. На большие расстояния вирус распространяется главным образом при перемещении инфицированных растений, особенно при пересадке томатов. Поскольку для проявления симптомов заболевания может потребоваться до 3 недель, зараженные растения без симптомов могут быть неосознанно перенесены. Вирус также может переноситься на большие расстояния белокрылками-переносчиками вируса, которые переносятся на томатах или других растениях (например, декоративных) или при сильном ветре, ураганах или тропических штормах.

Управление
Быстрые и точные тесты на вирус желтого скручивания листьев томатов доступны в Калифорнийском университете в Дэвисе и CDFA. Эти тесты могут быть проведены менее чем за 24 часа. Любой, кто обнаружит помидоры с симптомами, подобными TYLC, может связаться со своим окружным консультантом по сельскому хозяйству Робертом Л. Гилбертсоном из Калифорнийского университета в Дэвисе или Тонгьян Тянем из CDFA.

Стратегии эффективного лечения этого заболевания включают:

Перед посадкой
Выберите сорта, устойчивые к TYLCV.
Используйте трансплантаты, свободные от вирусов и белокрылки.
НЕ ввозите трансплантаты томатов (или любых других потенциальных хозяев белокрылки) из районов, о которых известно, что они заражены вирусом (Флорида, Джорджия и Техас в США; и Мексика).
В течение вегетационного периода
Высаживайте сразу после любого периода без помидоров или настоящей зимы.
Избегайте посадки новых культур рядом со старыми полями (особенно с растениями, инфицированными TYLCV).
Борьба с БЕЛОКРЫЛКАМИ.
Накройте растения плавающими рядками из мелкой сетки (Агрил или Агрибон) для защиты от нашествия белокрылки.
Зараженные растения-изгои, когда частота вирусной инфекции невелика.
Практикуйте надлежащую борьбу с сорняками на полях и вокруг них, насколько это возможно.
После окончания вегетационного периода
Убирают и уничтожают старые пожнивные остатки и волонтеры на региональной основе.
Добровольный или принудительный региональный период без хозяев в районах, где отсутствует настоящий зимний сезон (т.е. температуры достаточно низкие, чтобы препятствовать выращиванию сельскохозяйственных культур и выживанию белокрылки), может быть полезным инструментом управления. Культуры, которые будут включены в тот или иной регион, будут зависеть от агроэкосистемы.
"""
}

def main(page: ft.Page):
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_maximized=True
    page.bgcolor = '#2E3D49'
    page.fonts = {
        'Main': 'data/AnekLatin-VariableFont_wdth,wght.ttf',
    }
    def predictioning(path):
        loaded_model = load_model()
        formatted_image = format_image(path)
        result = make_prediction(loaded_model, formatted_image)
        text_message.value = f'{classes[np.argmax(result)]}'
        text_description.value = f'{disease_description[classes[np.argmax(result)]]}'
        page.update()


    def windspeed_change(e):
        try:
            value = float(windspeed_input.value)
            critical_wind = 7
            if value >= critical_wind:
                windspeed_warning.value = 'Жел тұрақсыз. Өскінге қолайсыз'
            else:
                windspeed_warning.value = 'Жел тұрақты. Өскінге қауіп жоқ'
        except Exception as e:
            windspeed_warning.value = 'Дұрыс ақпаратты енгізіңіз'
            print(e)
        page.update()
    

    def humidity_change(e):
        try:
            value = float(humidity_input.value)
            high_humidity = 70
            norm_humidity = 63
            low_humidity = 45
            if value < low_humidity:
                humidity_warning.value = 'Ылғалдылық жеткіліксіз деңгейде'
            elif value > high_humidity:
                humidity_warning.value = 'Ылғалдылық едәуір жоғары'
            elif value <= norm_humidity and value >= low_humidity:
                humidity_warning.value = 'Ылғалдылық нормаға сай'
            else:
                humidity_warning.value = 'Ылғалдылық жеткілікті деңгейде'
        except Exception as e:
            humidity_warning.value = 'Дұрыс ақпаратты енгізіңіз'
            print(e)
        page.update()


    def temperature_change(e):
        try:
            value = float(temperature_input.value)
            high_temperature = 36
            low_temperature = 14
            if value >= high_temperature:
                temperature_warning.value = 'Температура тым ыстық. Су құюдың жиілігін арттыру қажет.'
            elif value > low_temperature:
                temperature_warning.value = 'Температура қолайлы. Суды қалыпты мөлшерде кұю қажет.'
            else:
                temperature_warning.value = 'Температуры төмен. Су құюды қажет етпейді.'
        except Exception as e:
            temperature_warning.value = 'Дұрыс ақпаратты енгізіңіз'
            print(e)
        page.update()


    def animate_scanner():
        scanner_animation.top = 456
        page.update()
    

    def animate_scanner_2(e):
        scanner_animation.top = 173
        page.update()


    def file_chosen(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        new_image.src = e.files[0].path
        new_image.opacity = 100
        new_image.width = 401 - 36
        new_image.height = 401 - 36
        page.update()
        animate_scanner()
        predictioning(e.files[0].path)


    scanner_animation = ft.Container(
        border=ft.border.all(
            color=ft.colors.with_opacity(0.75, '#000000'),
            width=15,
        ),
        border_radius=ft.border_radius.all(25),
        width=333,
        height=48,
        top=173,
        left=174,
        animate_position=1500,
        on_animation_end=animate_scanner_2,
    )
    scanner = ft.Container(
        width=401,
        height=401,
        bgcolor=ft.colors.with_opacity(0.9, '#B9D2D2'),
        border_radius=15,
        top=137,
        left=140
    )
    filepicker = ft.FilePicker(
        on_result=file_chosen,
    )
    page.overlay.append(filepicker)
    choose_file_button = ft.ElevatedButton(
        text='Choose file',
        left=270,
        top=587,
        on_click=lambda _: filepicker.pick_files()
    )
    text_message = ft.Text(
        color=ft.colors.BLUE,
        weight=ft.FontWeight.W_700,
        size=25,
    )
    text_description = ft.Text(
        color=ft.colors.WHITE,
        size=20,
    )
    text_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        'Диагноз',
                        color=ft.colors.WHITE,
                        size=38.4,
                        font_family='Inter',
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Divider(
                color=ft.colors.BLACK,
                height=15,
            ),
            text_message,
            ft.Divider(
                color=ft.colors.BLACK,
                height=25,
            ),
            text_description,
        ],
        scroll=ft.ScrollMode.ADAPTIVE,
        height=500,
    )
    text_column_outer_container = ft.Container(
        margin=ft.margin.only(right=50),
        bgcolor=ft.colors.with_opacity(0.75, '#62A388'),
        border_radius=25,
        border=ft.border.all(
            color='#B9D2D2',
            width=3
        ),
        width=500,
        content=text_column,
        left=670,
        top=125,
        padding=20,
    )
    new_image = ft.Image(
        opacity=0,
        border_radius=ft.border_radius.all(15),
        src="data/1.png",
        width=0,
        height=0,
        fit=ft.ImageFit.CONTAIN,
        top=scanner.top + 18,
        left=scanner.left + 18
    )
    scanner_stack = ft.Stack(
        [
            scanner,
            new_image,
            scanner_animation,
            ft.Container(
                image_src='data/1.png',
                top=155,
                left=158,
                width=70,
                height=70,
            ),
            ft.Container(
                image_src='data/2.png',
                left=453,
                top=155,
                width=70,
                height=70
            ),
            ft.Container(
                image_src='data/3.png',
                left=453,
                top=450,
                width=70,
                height=70
            ),
            ft.Container(
                image_src='data/4.png',
                left=158,
                top=450,
                width=70,
                height=70
            ),
            choose_file_button,
            text_column_outer_container,
            ft.Text(
                'DFAN-01',
                color='white',
                font_family='Main',
                size=50,
                bottom=0,
                right=20
            ),
        ]
    )

    humidity_warning = ft.Text(
        'Ақпаратты енгізіңіз',
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500,
    )
    temperature_warning = ft.Text(
        'Ақпаратты енгізіңіз',
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500
    )
    windspeed_warning = ft.Text(
        'Ақпаратты енгізіңіз',
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500,
    )
    instructions_text_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        'Ескерту',
                        size=45,
                        font_family='Inter',
                        color=ft.colors.WHITE,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Divider(
                color=ft.colors.BLACK,
                height=0,
            ),
            ft.Row(
                [
                    ft.Text(
                        'Ылғалдылық бойынша:',
                        size=25,
                        weight=ft.FontWeight.W_500,
                        color=ft.colors.WHITE,
                    )
                ]
            ),
            ft.Row(
                [
                    ft.Container(
                        content=humidity_warning,
                        border_radius=7,
                        padding=10,
                        border=ft.border.all(
                            color=ft.colors.BLACK,
                            width=2,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Divider(
                height=25,
                color=ft.colors.BLACK,
            ),
            ft.Row(
                [
                    ft.Text(
                        'Жел бойынша:',
                        size=25,
                        weight=ft.FontWeight.W_500,
                        color=ft.colors.WHITE,
                    )
                ]
            ),
            ft.Row(
                [
                    ft.Container(
                        content=windspeed_warning,
                        padding=10,
                        border_radius=7,
                        border=ft.border.all(
                            color=ft.colors.BLACK,
                            width=2,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Divider(
                height=25,
                color=ft.colors.BLACK,
            ),
            ft.Row(
                [
                    ft.Text(
                        'Су бойынша:',
                        size=25,
                        weight=ft.FontWeight.W_500,
                        color=ft.colors.WHITE,
                    )
                ]
            ),
            ft.Row(
                [
                    ft.Container(
                        content=temperature_warning,
                        padding=10,
                        border_radius=7,
                        border=ft.border.all(
                            color=ft.colors.BLACK,
                            width=2,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Divider(
                color=ft.colors.BLACK,
                height=25,
            )

        ],
        scroll=ft.ScrollMode.ADAPTIVE
    )
    instructions_text_column_container = ft.Container(
        margin=ft.margin.only(right=50),
        bgcolor=ft.colors.with_opacity(0.75, '#62A388'),
        border_radius=30,
        border=ft.border.all(
            color='#B9D2D2',
            width=6
        ),
        width=765,
        height=562,
        content=instructions_text_column,
        left=493,
        top=125,
        padding=37,
    )
    humidity_input = ft.TextField(
        width=113,
        height=24,
        text_size=12,
        bgcolor='#D9D9D9',
        border_radius=ft.border_radius.all(0),
        on_submit=humidity_change,
    )
    humidity_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        'Ылғалдылық',
                        size=22.39,
                        font_family='Inter',
                        color=ft.colors.WHITE,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    humidity_input,
                ],
                alignment=ft.MainAxisAlignment.CENTER),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )
    humidity_cont = ft.Container(
        bgcolor=ft.colors.with_opacity(0, 'yellow'),
        border=ft.border.all(
            color=ft.colors.WHITE,
            width=3,
        ),
        alignment=ft.alignment.center,
        left=143,
        top=145,
        width=245,
        height=147,
        content=humidity_column
    )
    temperature_input = ft.TextField(
        width=113,
        height=24,
        text_size=12,
        bgcolor='#D9D9D9',
        border_radius=ft.border_radius.all(0),
        on_submit=temperature_change,
    )
    temperature_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        'Температура',
                        size=22.39,
                        font_family='Inter',
                        color=ft.colors.WHITE,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    temperature_input,
                ],
                alignment=ft.MainAxisAlignment.CENTER),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )
    temperature_cont = ft.Container(
        bgcolor=ft.colors.with_opacity(0, 'yellow'),
        border=ft.border.all(
            color=ft.colors.WHITE,
            width=3,
        ),
        alignment=ft.alignment.center,
        left=143,
        top=337,
        width=245,
        height=147,
        content=temperature_column,
    )
    windspeed_input = ft.TextField(
        width=113,
        height=24,
        text_size=12,
        bgcolor='#D9D9D9',
        border_radius=ft.border_radius.all(0),
        on_submit=windspeed_change,
    )
    windspeed_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        'Жел жылдамдығы',
                        size=22.39,
                        font_family='Inter',
                        color=ft.colors.WHITE,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row(
                [
                    windspeed_input,
                ],
                alignment=ft.MainAxisAlignment.CENTER),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    )
    windspeed_cont = ft.Container(
        bgcolor=ft.colors.with_opacity(0, 'yellow'),
        border=ft.border.all(
            color=ft.colors.WHITE,
            width=3,
        ),
        alignment=ft.alignment.center,
        left=143,
        top=518,
        width=245,
        height=147,
        content=windspeed_column,
    )
    instructions_stack = ft.Stack(
        [
            instructions_text_column_container,
            humidity_cont,
            temperature_cont,
            windspeed_cont,
            ft.Text(
                'DFAN-01',
                color='white',
                font_family='Main',
                size=50,
                bottom=0,
                right=20
            ),
        ]
    )

    tabs = ft.Tabs(
        selected_index=1,
        animation_duration=300,
        tabs=[
            ft.Tab(
                'Scanner',
                content=scanner_stack,
            ),
            ft.Tab(
                'Instructions',
                content=instructions_stack,
            ),
        ],
        label_color=ft.colors.BLUE,
        unselected_label_color=ft.colors.WHITE,
        indicator_tab_size=False,
        indicator_padding=0,
        divider_color=ft.colors.with_opacity(0, '#000000'),
        indicator_border_side=ft.colors.with_opacity(0, '#000000'),
        expand=1,
    )
    page.add(tabs)
    page.update()


ft.app(target=main)