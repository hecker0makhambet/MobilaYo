import flet as ft
import numpy as np
import json
from PIL import Image
import numpy as np
from keras.models import model_from_json
from keras.utils import img_to_array
import os

lang = 1 # English, 0 - Kazakh

low_temperature_warning = [
    'Температуры төмен. Су құюды қажет етпейді.',
    'The temperature is low. Does not require pouring water.'
]
suitable_temperature_warning = [
    'Температура қолайлы. Суды қалыпты мөлшерде кұю қажет.',
    'The temperature is suitable. A moderate amount of water is required.'
]
high_temperature_warning = [
    'Температура тым ыстық. Су құюдың жиілігін арттыру қажет.',
    'The temperature is too high. It is necessary to increase the frequency of watering.'
]
sufficient_humidity = [
    'Ылғалдылық жеткілікті деңгейде',
    'Humidity at a sufficient level'
]
humidity_is_normal = [
    'Ылғалдылық нормаға сай',
    'Humidity is normal'
]
humidity_is_high = [
    'Ылғалдылық едәуір жоғары',
    'Humidity is significantly high'
]
insufficient_humidity = [
    'Ылғалдылық жеткіліксіз деңгейде',
    'Insufficient humidity'
]
enter_the_valid_data = [
    'Дұрыс ақпаратты енгізіңіз',
    'Enter the valid data'
]
wind_is_stable = [
    'Жел тұрақты. Өскінге қауіп жоқ',
    'The wind is constant. There is no danger to growth'
]
wind_is_unstable = [
    'Жел тұрақсыз. Өскінге қолайсыз',
    'The wind is unstable. Unfavorable for growth'
]
windspeed = [
    'Жел жылдамдығы',
    'Wind speed'
]
temperature = [
    'Температура',
    'Temperature'
]
humidity = [
    'Ылғалдылық',
    'Humidity'
]
by_water = [
    'Су бойынша:',
    'By water:'
]
by_wind = [
    'Жел бойынша:',
    'By wind:'
]
by_humidity = [
    'Ылғалдылық бойынша:',
    'By humidity:'
]
warning = [
    'Ескерту',
    'Warning'
]
enter_the_data = [
    'Ақпаратты енгізіңіз',
    'Enter the data'
]
diagnosis = [
    'Диагноз', 'Diagnosis' 
]


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

disease_description_english = {
    'Potato___Early_blight': """Symptoms and Signs
Early blight is primarily a disease of stressed or senescing plants. Symptoms appear first on the oldest foliage. Affected leaves develop circular to angular dark brown lesions 0.12 to 0.16 inch (3–4 mm) in diameter. Concentric rings often form in lesions to produce characteristic target-board effect. Severely infected leaves turn yellow and drop. Infected tubers show a brown, corky dry rot.

Comments on the Disease
Between crops, the early blight fungus can overwinter on potato refuse in the field, in soil, on tubers, and on other solanaceous plants. Infection occurs when spores of the fungus come in contact with susceptible leaves and sufficient free moisture is present. Spore germination and infection are favored by warm weather and wet conditions from dew, rain, or sprinkler irrigation. Alternately, wet and dry periods with relatively dry, windy conditions favor spore dispersal and disease spread. Tubers can be infected as they are lifted through the soil at harvest. If sufficient moisture is present, spores germinate and infect the tubers.

Management
Early blight can be minimized by maintaining optimum growing conditions, including proper fertilization, irrigation, and management of other pests. Grow later maturing, longer season varieties. Fungicide application is justified only when the disease is initiated early enough to cause economic loss. Watch for disease symptoms during routine monitoring, and keep records of your results. When justified, apply fungicides as soon as symptoms appear; continued protection requires application at 7- to 10-day intervals.
""", # https://ipm.ucanr.edu/agriculture/potato/early-blight/
    'Corn_(maize)___Common_rust_': """Common corn rust, caused by the fungus Puccinia sorghi, is the most frequently occurring of the two primary rust diseases of corn in the U.S., but it rarely causes significant yield losses in Ohio field (dent) corn. Occasionally field corn, particularly in the southern half of the state, does become severely affected when weather conditions favor the development and spread of the rust fungus. Sweet corn is generally more susceptible than field corn. In years with exceptionally cool summers, and especially on late-planted fields or sweet corn, yield losses may occur when the leaves at and above the ears become severely diseased before grain fill is complete.

Symptoms

Although a few rust pustules can always be found in corn fields throughout the growing season, symptoms generally do not appear until after tasseling. These can be easily recognized and distinguished from other diseases by the development of dark, reddish-brown pustules (uredinia) scattered over both the upper and lower surfaces of the corn leaves (Fig. 1). These pustules may appear on any above ground part of the plant, but are most abundant on the leaves. Pustules appear oval to elongate in shape, are generally small, less than 1/4 inch long, and are surrounded by the leaf epidermal layer, where it has broken through. If infections occur while the leaves are still in the whorl, these pustules may develop in bands across the surface as the leaf expands in size.

Disease Cycle and Epidemiology
Unlike most other foliar diseases of corn, this rust fungus does not overwinter in crop residue. Spores must be blown northward during the growing season (from mid-June to mid-July) from subtropical and tropical regions where this fungus survives on corn or wood sorrel, the alternate host. Young leaves are generally more susceptible to infection than older leaves. Rust development and spread are favored by prolonged periods of cool temperatures ranging from 60° to 74°F and high relative humidity. Under these conditions, pustules develop on susceptible corn hybrids and sweet corn varieties within 7 days after infection.

Occasionally, chlorosis and death of the leaves and leaf sheaths occur when infections are severe. Uredospores produced during the season are wind disseminated, spreading the pathogen to new leaves, plants and fields. As the corn plant matures pustules turn brownish-black in color due to the development of darker pigmented telia that replace uredinia and produce teliospores. In tropical regions teliospores infect the alternate host, wood sorrel (Oxalis species), however, in temperate areas such as Ohio and other states in the U.S. cornbelt, the fungus does not infect wood sorrel and the teliospores have no real epidemiological significance (do not contribute to the disease cycle).

Management
Although rust is frequently found on corn in Ohio, very rarely has there been a need for fungicide applications. This is due to the fact that there are highly resistant field corn hybrids available and most possess some degree of resistance. However, popcorn and sweet corn can be quite susceptible. In seasons where considerable rust is present on the lower leaves prior to silking and the weather is unseasonably cool and wet, an early fungicide application may be necessary for effective disease control. Numerous fungicides are available for rust control. Consult your local county extension office and C.O.R.N. website for the latest recommendations for efficacy. Always read the fungicide label for rates and application timing.
""", # https://ohioline.osu.edu/factsheet/plpath-cer-02
    'Tomato___Early_blight': """Early blight -Alternaria linariae (=A. tomatophila; formerly known as A. solani)

Infections begin as small brown spots on older leaves that quickly enlarge. A yellow halo usually surrounds the lesions. The lesions develop a "bulls-eye" pattern of concentric rings that can be seen with a hand lens. Individual lesions enlarge and coalesce and can kill entire leaves. The disease can also move to stems and fruits and produce dark lesions.
This is a very common foliar disease of tomato plants that can result in defoliation and reduced yields. It can also infect eggplant. The fungus overwinters in soil and on plant debris.  It can also be transmitted on seeds and transplants. Early blight is typically splash

Management
Provide adequate spacing to increase air circulation and remove all suckers that emerge from the plant base
Monitor transplants carefully for signs of this disease.
Keep plants well mulched to minimize soil splashing.
Water your plants around their base. Avoid wetting foliage.
Prune off the lowest 3-4 leaf branches once plants are well established and starting to develop fruits.
Remove infected leaves during the growing season and remove all infected plant parts at the end of the season.
Apply a synthetic fungicide or an organic fungicide (fixed copper) according to label directions, early in the season, when symptoms appear to slow the spread of the disease. This may be helpful where the disease causes severe blighting each year leading to reduced yields.
Diseased plant parts can be shredded and composted if "hot composting" techniques are used (pile temperatures should exceed 120° F throughout and piles should be turned two to three times).
""", # https://extension.umd.edu/resource/early-blight-tomatoes
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': """Symptoms and Signs
Tomato yellow leaf curl is a disease of tomato caused by Tomato yellow leaf curl virus. In March 2007, it was identified for the first time in California and currently has a limited distribution. An educational brochure (PDF) was created at that time and is available to print.

Infected tomato plants initially show stunted and erect or upright plant growth; plants infected at an early stage of growth will show severe stunting. However, the most diagnostic symptoms are those in leaves.

Leaves of infected plants are small and curl upward; and show strong crumpling and interveinal and marginal yellowing. The internodes of infected plants become shortened and, together with the stunted growth, plants often take on a bushy appearance, which is sometimes referred to as 'bonsai' or broccoli'-like growth. Flowers formed on infected plants commonly do not develop and fall off (abscise). Fruit production is dramatically reduced, particularly when plants are infected at an early age, and it is not uncommon for losses of 100% to be experienced in fields with heavily infected plants.

Comments on the Disease
Tomato yellow leaf curl virus is undoubtedly one of the most damaging pathogens of tomato, and it limits production of tomato in many tropical and subtropical areas of the world. It is also a problem in many countries that have a Mediterranean climate such as California. Thus, the spread of the virus throughout California must be considered as a serious potential threat to the tomato industry.

There are a number of factors why it has not yet spread to all the major tomato-producing areas of California, including the Sacramento and San Joaquin valleys. First, its vector, Bemisia whitefly species are not typically found in these tomato-producing areas because it is intolerant of winter temperatures there. Second, the Central Valley's winter season provides a 'natural' tomato-free period, which usually goes from late November through early February. Although the virus can infect other plants, tomato is the host in which it builds-up most quickly. Thus, by having an annual 'tomato-free period', it is likely that the amount of viral inoculum (as well as whitefly populations) will be significantly reduced by the time the tomato planting season starts again in late winter-early spring. This would mean that, even if the virus is able to overwinter, it may take a long time to reach levels that cause economic damage.

Tomato yellow leaf curl virus is a geminivirus (family Geminiviridae). Although it can infect a relatively wide range of plant species, tomato is the host to which the virus is best adapted and that facilitates the build-up of the virus to high incidences in the field. Other hosts include solanaceous crops, which may develop symptomless infections, and weeds (e.g., nightshade and jimsonweed).

In addition, the virus causes leaf curl in certain varieties of common bean (Phaseolus vulgaris) and the ornamental plant lisianthus (Eustoma grandiflorum). A range of weeds from other families can be infected by this virus, but most of these do not develop obvious disease symptoms. It is not known how well whiteflies acquire the virus from symptomless hosts. However, it has been hypothesized that these hosts serve as a 'bridge' for the virus in the absence of tomato crops, and that perennial weeds help allow the virus to become permanently established.

The primary way the virus is spread short distances is by Bemisia whitefly species. Over long distance, the virus is primarily spread through the movement of infected plants, especially tomato transplants. Because it can take up to 3 weeks for disease symptoms to develop, infected symptomless plants could be unknowingly transported. The virus also can be moved long distance by virus-carrying whiteflies that are transported on tomatoes or other plants (e.g., ornamentals) or via high winds, hurricanes, or tropical storms.

Management
Rapid and precise tests for Tomato yellow leaf curl virus are available at UC Davis and CDFA. These tests can be carried out in less than 24 hours. Anyone finding tomatoes with TYLC-like symptoms can contact their county farm advisor, Robert L. Gilbertson at UC Davis, or Tongyan Tian at CDFA.

Strategies to effectively manage the disease include:

Before Planting
Select TYLCV-resistant varieties.
Use virus- and whitefly-free transplants.
DO NOT import tomato (or any potential whitefly host) transplants from areas known to have the virus (Florida, Georgia and Texas in the U.S.; and Mexico).
During the Growing Season
Plant immediately after any tomato-free period or true winter season.
Avoid planting new fields near older fields (especially those with TYLCV-infected plants).
Manage WHITEFLIES.
Cover plants with floating row covers of fine mesh (Agryl or Agribon) to protect from whitefly infestations.
Rogue diseased plants when incidence of virus infection is low.
Practice good weed management in and around fields to the extent feasible.
After the Growing Season
Remove and destroy old crop residue and volunteers on a regional basis.
A voluntary or enforced regional host-free period in areas lacking a true winter season (i.e., temperatures low enough to prevent crop cultivation and whitefly survival) might be a useful management tool. The crops to be included in a region will depend on the agroecosystem.
""", # https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/
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
        text_description.value = f'{disease_description_english[classes[np.argmax(result)]]}'
        page.update()


    def windspeed_change(e):
        try:
            value = float(windspeed_input.value)
            critical_wind = 7
            if value >= critical_wind:
                windspeed_warning.value = wind_is_unstable[lang]
            else:
                windspeed_warning.value = wind_is_stable[lang]
        except Exception as e:
            windspeed_warning.value = enter_the_valid_data[lang]
            print(e)
        page.update()
    

    def humidity_change(e):
        try:
            value = float(humidity_input.value)
            high_humidity = 70
            norm_humidity = 63
            low_humidity = 45
            if value < low_humidity:
                humidity_warning.value = insufficient_humidity[lang]
            elif value > high_humidity:
                humidity_warning.value = humidity_is_high[lang]
            elif value <= norm_humidity and value >= low_humidity:
                humidity_warning.value = humidity_is_normal[lang]
            else:
                humidity_warning.value = sufficient_humidity[lang]
        except Exception as e:
            humidity_warning.value = enter_the_valid_data[lang]
            print(e)
        page.update()


    def temperature_change(e):
        try:
            value = float(temperature_input.value)
            high_temperature = 36
            low_temperature = 14
            if value >= high_temperature:
                temperature_warning.value = high_temperature_warning[lang]
            elif value > low_temperature:
                temperature_warning.value = suitable_temperature_warning[lang]
            else:
                temperature_warning.value = low_temperature_warning[lang]
        except Exception as e:
            temperature_warning.value = enter_the_valid_data[lang]
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
                        diagnosis[lang],
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
        enter_the_data[lang],
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500,
    )
    temperature_warning = ft.Text(
        enter_the_data[lang],
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500
    )
    windspeed_warning = ft.Text(
        enter_the_data[lang],
        color=ft.colors.WHITE,
        size=15,
        weight=ft.FontWeight.W_500,
    )
    instructions_text_column = ft.Column(
        [
            ft.Row(
                [
                    ft.Text(
                        warning[lang],
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
                        by_humidity[lang],
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
                        by_wind[lang],
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
                        by_water[lang],
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
                        humidity[lang],
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
                        temperature[lang],
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
                        windspeed[lang],
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