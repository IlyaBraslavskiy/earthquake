# earthquake

## Технические ноутбки/файлы

| Файл | Описание |
|------|----------|
|dataset_from_data.ipynb|Начальный файл csv перегонятеся в pd, время и координаты перводятся в удобный формат. Подробнее в шапке ноутбука. Результат работы  work_catalog_pd. Далее как стартовая точка работы других ноутбуков испольузется этот файл.|
|scripts/make_grid.py|Дасет разбивается на ячейки по координатам и по времени. В каждую ячейку в такой матрице записывается датасет из землетрясений, удовлетворяющих этой ячейке. Это необходимо для расчета временных рядов предвестников по пространственной сетке, т.е. для каждой ячейки свой временной ряд. В ноутбуках при необходимости импортируется как from scripts import make_grid as mg| 

## Ноутбуки/файлы про аппроксимацию пуассоновской модели
| Файл | Описание |
|------|----------|
|bayes_poisson_with_covariates.ipynb|Ноутбук в котором есть пример расчета фичей с помощью аппроксимации пуассоновской модели и ее вывод. Иcпользует scripts/poisson_covariance as и scripts/make_grid|
|scipts/poisson_covariance|Функции с расчетом всякого для аппроксимации пуассоновской модели, пример использования выше|
|predictive_selector_poisson_cov.ipynb|Пример использования предиктивного селектора на фичах. *Предиктивный селектор* на github не выкладывается, он есть в соотвествующей папке которую я скидывал.|

## Ноутбуки/файлы про RTL
| Файл | Описание |
|------|----------|
|RTl.ipynb| Непараметрическая модель RTL предвестника, его описание, код и пример использования|
|RTL_at_clasters.ipynb|Пример рассчета прекурсора RTL в кластерах

## Ноутбуки/файлы с кластеризацией
| Файл | Описание |
|------|----------|
|test_reg.ipynb| Ноутбук в котором примеры использования моделей кластеризации (GMM и по направлению) которые за коденв в файле clusterization. *vbgmm* это gmm модель, *variational lmm* это смесь регрессий|
|/clusterization| Папка в которой код различных методов кластеризации. Демоснтарция см. выше|

## Текст
| Файл | Описание |
|------|----------|
|/thesis|pdf и tex исходники диплома| 