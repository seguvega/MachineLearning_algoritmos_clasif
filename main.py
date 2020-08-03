from flask import Flask, render_template,request
from flask import Flask, render_template,request,Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from algoritmos import (regresion_logistica,resultado_variables,metricas_reg_log,m_soporte_vectorial,red_neuronal,knn,random,nbayes,cargar_iris,
dispersion_iris,tiempos,pred_real_iris,dispersion_externo)

app = Flask(__name__)
x_train, x_test, y_train, y_test, x2_train, x2_test, y2_train, y2_test=resultado_variables()
iris,iris2=cargar_iris()

@app.route('/plot_ext.png')
def plot2():
    fig = dispersion_externo()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(),mimetype='image/png')

@app.route('/plot_iris.png')
def plot():
    fig = dispersion_iris()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(),mimetype='image/png')

@app.route('/')
def iris():
    pred_reg_log_iris, accuracy_log, recall_log, pres_log, f1_log,tiempo_reg = regresion_logistica(x_train, y_train, x_test,y_test)
    pred_soporte_vectorial_iris, accuracy_vec, recall_vec, pres_vec, f1_vec,tiempo_vec = m_soporte_vectorial(x_train, y_train,
                                                                                                  x_test, y_test)
    pred_red_neurona_iris, accuracy_neu, recall_neu, pres_neu, f1_neu,tiempo_neu = red_neuronal(x_train, y_train, x_test, y_test)
    pred_knn_iris, accuracy_knn, recall_knn, pres_knn, f1_knn,tiempo_knn = knn(x_train, y_train, x_test, y_test)
    pred_ran_iris, accuracy_ran, recall_ran, pres_ran, f1_ran,tiempo_ran = random(x_train, y_train, x_test, y_test)
    pred_nb1, accuracy_nb, recall_nb, pres_nb, f1_nb,tiempo_nb = nbayes(x_train, y_train, x_test, y_test)

    metricas_rl=metricas_reg_log(accuracy_log, recall_log, pres_log, f1_log)
    metricas_sv = metricas_reg_log(accuracy_vec, recall_vec, pres_vec, f1_vec)
    metricas_rn = metricas_reg_log(accuracy_neu, recall_neu, pres_neu, f1_neu)
    metricas_knn = metricas_reg_log(accuracy_knn, recall_knn, pres_knn, f1_knn)
    metricas_ran = metricas_reg_log(accuracy_ran, recall_ran, pres_ran, f1_ran)
    metricas_nb = metricas_reg_log(accuracy_nb, recall_nb, pres_nb, f1_nb)

    tiempos1=tiempos(tiempo_knn,tiempo_ran,tiempo_nb,tiempo_reg,tiempo_vec,tiempo_neu)
    reg_log_iris,s_vectorial,red,knn1,ran,nb=pred_real_iris(y_test,pred_reg_log_iris,pred_soporte_vectorial_iris,
                                                              pred_red_neurona_iris,pred_knn_iris,pred_ran_iris,pred_nb1)

    item="nav-item active"
    item2 = "nav-item"
    item3 = "nav-item"
    return render_template('menu.html',item=item,item2=item2,item3=item3,pred=pred_reg_log_iris,metricas_rl=metricas_rl,metricas_sv=metricas_sv,
                           metricas_rn=metricas_rn,metricas_knn =metricas_knn,  metricas_ran=metricas_ran,metricas_nb=metricas_nb,tiempos=tiempos1,
                           reg_log_iris=reg_log_iris,s_vectorial=s_vectorial,red=red,knn=knn1,ran=ran,nb=nb)


@app.route('/externo')
def ecoli():
    pred_reg_log_ecoli, accuracy_log2, recall_log2, pres_log2, f1_log2,tiempo_reg2 = regresion_logistica(x2_train, y2_train, x2_test,
                                                                                        y2_test)

    pred_soporte_vectorial_ecoli, accuracy_vec2, recall_vec2, pres_vec2, f1_vec2,tiempo_vec2 = m_soporte_vectorial(x2_train, y2_train,
                                                                                                  x2_test, y2_test)
    pred_red_neurona_ecoli, accuracy_neu2, recall_neu2, pres_neu2, f1_neu2,tiempo_neu2 = red_neuronal(x2_train, y2_train, x2_test, y2_test)
    pred_knn_ecoli, accuracy_knn2, recall_knn2, pres_knn2, f1_knn2,tiempo_knn2 = knn(x2_train, y2_train, x2_test, y2_test)
    pred_ran_ecoli, accuracy_ran2, recall_ran2, pres_ran2, f1_ran2,tiempo_ran2 = random(x2_train, y2_train, x2_test, y2_test)
    pred_nb1_ecoli, accuracy_nb2, recall_nb2, pres_nb2, f1_nb2,tiempo_nb2 = nbayes(x2_train, y2_train, x2_test, y2_test)

    metricas_rl2 = metricas_reg_log(accuracy_log2, recall_log2, pres_log2, f1_log2)
    metricas_sv2 = metricas_reg_log(accuracy_vec2, recall_vec2, pres_vec2, f1_vec2)
    metricas_rn2 = metricas_reg_log(accuracy_neu2, recall_neu2, pres_neu2, f1_neu2)
    metricas_knn2 = metricas_reg_log(accuracy_knn2, recall_knn2, pres_knn2, f1_knn2)
    metricas_ran2 = metricas_reg_log(accuracy_ran2, recall_ran2, pres_ran2, f1_ran2)
    metricas_nb2 = metricas_reg_log(accuracy_nb2, recall_nb2, pres_nb2, f1_nb2)

    tiempos1 = tiempos(tiempo_knn2, tiempo_ran2, tiempo_nb2, tiempo_reg2, tiempo_vec2, tiempo_neu2)
    reg_log_iris2, s_vectorial2, red2, knn12, ran2, nb2 = pred_real_iris(y2_test, pred_reg_log_ecoli,
                                                                   pred_soporte_vectorial_ecoli,
                                                                   pred_red_neurona_ecoli, pred_knn_ecoli, pred_ran_ecoli,
                                                                   pred_nb1_ecoli)
    item = "nav-item"
    item2 = "nav-item active"
    item3 = "nav-item"

    return render_template('haberman.html', item=item, item2=item2, item3=item3, pred=pred_reg_log_ecoli,
                           metricas_rl=metricas_rl2, metricas_sv=metricas_sv2,
                           metricas_rn=metricas_rn2, metricas_knn=metricas_knn2, metricas_ran=metricas_ran2,
                           metricas_nb=metricas_nb2,tiempos=tiempos1,reg_log_iris=reg_log_iris2,s_vectorial=s_vectorial2,
                           red=red2,knn=knn12,ran=ran2,nb=nb2)

if __name__ == '__main__':
    app.run(debug=True,port=5000,use_reloader=False)