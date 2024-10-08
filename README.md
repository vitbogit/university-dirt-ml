# 2023 ML для анализа почвы со спутника (RU)

> ❤️ Дисклеймер   
Автор сам изучал ML и не имел поддержки преподавателя, этот код даже никто не проверил :(  
Работа уже на старте не имела возможности оказаться сильно полезной на практике: отсутствовали критические поля в тестовых данных, сами данные имели чрезвычайно малый объем. Поэтому
работа носила скорее следующий характер: "Если по итогам получится хоть сколько-то убедительный график, с которым можно пойти к инвесторам и сказать, что ML умеет анализировать почву, а что
там под капотом у этого графика не важно - это победа".     

В текущий момент анализ почвы во многих областях РФ производится по старинке: выкапывается земля, берутся образцы, делается анализ на содержание тех или иных веществ. 
Такой подход проверен временем и работает, позволяет делать выводы о пригодности почвы к засеиванию. Однако у такого подхода есть и существенный недостаткок - этот процесс
очень дорого повторять раз за разом на различных территориях, а также это требует больших временных затрат.

Спутники регулярно делают снимки планеты и могут предоставить собранную информацию. Так вот целью моей работы является попытка использовать информацию со спутников для того, чтобы
заменить ручной труд, обучив модель предсказывать содержание в почве полезных веществ на основании цветовых данных, считанных датчиками спутника.

Моя работа не отягащена вынесением моделью вердикта о пригодности почвы, а лишь обучается по тестовым данным предсказывать содержание нескольких групп полезных веществ и строить графики.
Так в чем же в итоге польза от программы? Если модель окажется способной "неплохо" видеть закономерности в данных со спутника и отображать их, то это и будет первым шагом в сторону
автоматизации проверки почв.

К сожалению, тестовых данных было крайне мало, но по анализу некоторых веществ из почвы получилось получить подобные результаты:  

![image](https://github.com/vitbogit/university-dirt-ml/assets/61887732/e0c62a4b-a3b4-4550-a764-01133b53f277)
- гумус - это такое вещество в почве
- 4 канал - один из датчиков спутника
- синее - модель на этом училась
- серое - модель должна была это предсказать
- красные кружки - модель предсказала это вместо серого
- красная линяя - усреднее красных кружков линеей

Что мы видим:
1) модель "увидела", что при понижении показателей датчика ниже 2000 содержание гумуса возрастает;
2) при повышении показателей датчика выше 2000 что-то сложно сказать о содержании гумуса, и модель ожидаемо плохо справилась.
   
![aaaa](https://github.com/vitbogit/university-dirt-ml/assets/61887732/30c382ba-ca72-4ac0-b8ac-88c106d1634b)


Что касается первого пункта - мы частично достигли своей цели, благодаря модели экологи смогли бы автоматически довольно точно "предсказать" некоторые показатели почвы в определенных условиях.
Что касается второго пункта - необходимо понимать, что на скриншоте представлен показатель лишь одного из почти десятка!!! веществ в почве, имеющих высокое значение для ее качества, а также по этому веществу 
на скриншоте представлен анализ лишь по одному из более чем десятка!!! датчиков спутника. Модель строилась так, чтобы учиться на всех веществах и по всем датчикам, но графики строятся по каждому веществу и датчику отдельно. Почему выбран такой формат вывода? Потому что, к сожалению, в тестовых данных не дается пометки о статусе почвы - пригодная / не пригодная, и решилась по сути не задача классификации, а задача регрессии.

Еще несколько примеров вывода программы на других веществах и датчиков:

![1](https://github.com/vitbogit/university-dirt-ml/assets/61887732/2c0d4225-cc91-42ed-8d87-a7feb8fb7510)

![kall](https://github.com/vitbogit/university-dirt-ml/assets/61887732/f27489f4-358b-495f-81a9-a7d4c651e6f3)


Выводы:
- с подобным подходом модель способна с некоторой точностью предсказывать содержание полезных веществ в почве и генерировать наглядные отчеты, что позволяет считать цель работы достигнутой;
- отсутствие возможности обучить модель классифицировать почву как носящую тот или иной статус из-за недостаточных тестовых данных, а также в принципе малый объем тестовых данных принудили автора к попытке "предсказать все сразу, и ничего одновременно", еще и имея для этого мизерный объем информации, из-за чего программа незаслуженно теряет возможность быть практически полезной;
- так как имеем задачу регрессии, а не классификации, то без знаний предметной области вообще невозможно судить о полезности программы по ее предсказаниям чисел: с огромной долей вероятности эти "закономерности" на графиках представленные моделью могут буквально обозначать: "Если на фото со спутника что-то цвета озера, это скорее всего не трава...", - в таком случае польза от данной модели почти полностью теряется.

Не стоит унывать, экологи! Это всего лишь до неприличного простая работа, однако и она не лишена положительного показательного момента: показан потенциал для исследований в этом направлении, а это главное. 
