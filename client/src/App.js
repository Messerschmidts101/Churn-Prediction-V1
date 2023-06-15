import Button from './Components/Button'
import Header1 from './Components/Header1'
import Header2 from './Components/Header2'
import Input from './Components/Input'
import Form from './Components/Form'
import Label from "./Components/Label"
import InputFeature from './Components/InputFeature'
import { useState, useEffect } from 'react'
import Highcharts from 'highcharts'
import HighchartsReact from 'highcharts-react-official'
import './CSS/main.css'

function App() {

    const [feature_list, setFeatureList] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setFeatureList(data)
            }
        )
    }, [])
    let [feature_post, setFeaturePost] = useState([{}])
    let [series, setSeries] = useState([{}])
    // useEffect(() => {
    //     fetch("http://localhost:8080/prediction", {
    //         method: 'GET',
    //         mode: 'cors',
    //         body: JSON.stringify()
    //     })
    // }, [])

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("[type=number]");
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleChurn = () => {
        let feature_post_temp = []
        let inputFields = document.querySelectorAll("[type=number]");
        for(let i = 0; i < inputFields.length; i++) {
            let nameID = inputFields[i].id.split(" ")[0]
            let value = inputFields[i].value
            feature_post_temp.push({
                nameID: value
            })
        }
        setFeaturePost(feature_post_temp)

        fetch("http://localhost:8080/prediction", {
            method: "POST",
            mode: "cors",
            body: JSON.stringify(feature_post)
        })
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name).value = ""
        })
    }

    const options = {
        series: series,
        title: {
            text: "Churn of Customer"
        },
        chart: {
            type: "line"
        },
        xAxis: {
            categories: []
        }
    }
    
    return (
        <main className='main'>
            <section id="hero" className='d-flex align-items-center'>
                <div className='title text-center'>
                    <p className='title-heading' children={"Churn Prediction"}/>
                    <p className='title-subheading' children={"using Random Forest Classifier"}/>
                    <a href="#predict"><button className='btn' children={"Try the model"}/></a>
                    
                </div>
            </section>
            
            <section id='predict'>
                <div className='nav'>
                        <p className='nav-heading' children={"Churn Prediction"}/>
                        <p className='nav-subheading' children={"using Random Forest Classifier"}/>
                </div>
                
                <section id="input" className='container'>
                    <div className='inputValues'>
                        <Header1 className={"text-center label1"}  children={"Input Your Values"} />
                        <div className='d-flex flex-row align-items-center text-center'>
                            <div className='form-floating w-1100'>
                                <Input id={"pasteData"}  className={"form-control d-flex input-row"} type="text" placeholder="Paste Excel row here" />
                                <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                            </div>
                            <Button className={"d-flex inline scatter-btn"} children={"Scatter Values"} onClick={handleScatter} />
                        </div>

                        <Header2 className={"mt-3"} children={"Or input below each value individually"}></Header2>
                        <Form id={"predictionForm"} className={"form row"} action={"prediction"} method={"post"}>
                            <table>
                                <tr>
                                    <td className='form-floating'>
                                        <Input id={"state"} className={"form-control"} type={"number"} placeholder={"state"} required={"required"} />
                                        <Label forLabel={"state"} children={"State"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"account_length"} className={"form-control"} type={"number"} placeholder={"account_length"} required={"required"} />
                                        <Label forLabel={"account_lengthLbl"} children={"Account Length"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"area_code"} className={"form-control"} type={"number"} placeholder={"area_code"} required={"required"} />
                                        <Label forLabel={"area_codeLbl"} children={"Area Code"} />
                                    </td>
                                </tr>
                                <tr>
                                    <td className='form-floating'>
                                        <Input id={"international_plan"} className={"form-control"} type={"number"} placeholder={"international_plan"} required={"required"} />
                                        <Label forLabel={"international_planLbl"} children={"International Plan"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"voice_mail_plan"} className={"form-control"} type={"number"} placeholder={"voice_mail_plan"} required={"required"} />
                                        <Label forLabel={"voice_mail_planLbl"} children={"Voice Mail Plan"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"number_vmail_messages"} className={"form-control"} type={"number"} placeholder={"number_vmail_messages"} required={"required"} />
                                        <Label forLabel={"number_vmail_messagesLbl"} children={"Number of Voice Mails"} />
                                    </td>
                                </tr>
                                <tr>
                                    <td className='form-floating'>
                                        <Input id={"total_day_minutes"} className={"form-control"} type={"number"} placeholder={"total_day_minutes"} required={"required"} />
                                        <Label forLabel={"total_day_minutesLbl"} children={"Total Minutes (Day)"} />                                    
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_day_calls"} className={"form-control"} type={"number"} placeholder={"total_day_calls"} required={"required"} />
                                        <Label forLabel={"total_day_callsLbl"} children={"Total Calls (Day)"} />                                        
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_day_charge"} className={"form-control"} type={"number"} placeholder={"total_day_charge"} required={"required"} />
                                        <Label forLabel={"total_day_chargeLbl"} children={"Total Charges (Day)"} />
                                    </td>
                                </tr>
                                <tr>
                                    <td className='form-floating'> 
                                        <Input id={"total_eve_minutes"} className={"form-control"} type={"number"} placeholder={"total_eve_minutes"} required={"required"} />
                                        <Label forLabel={"total_eve_minutesLbl"} children={"Total Minutes (Evening)"} />                                       
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_eve_calls"} className={"form-control"} type={"number"} placeholder={"total_eve_calls"} required={"required"} />
                                        <Label forLabel={"total_eve_callsLbl"} children={"Total Calls (Evening)"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_eve_charge"} className={"form-control"} type={"number"} placeholder={"total_eve_charge"} required={"required"} />   
                                        <Label forLabel={"total_eve_chargeLbl"} children={"Total Charge (Evening)"} />
                                    </td>
                                </tr>
                                <tr>                                    
                                    <td className='form-floating'>
                                        <Input id={"total_night_minutes"} className={"form-control"} type={"number"} placeholder={"total_night_minutes"} required={"required"} /> 
                                        <Label forLabel={"total_night_minutesLbl"} children={"Total Minutes (Night)"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_night_calls"} className={"form-control"} type={"number"} placeholder={"total_night_calls"} required={"required"} />
                                        <Label forLabel={"total_night_callsLbl"} children={"Total Calls (Night)"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_night_charge"} className={"form-control"} type={"number"} placeholder={"total_night_charge"} required={"required"} />
                                        <Label forLabel={"total_night_chargeLbl"} children={"Total Charge (Night)"} />
                                    </td>
                                </tr>
                                <tr>
                                    <td className='form-floating'>
                                        <Input id={"total_intl_minutes"} className={"form-control"} type={"number"} placeholder={"total_intl_minutes"} required={"required"} />
                                        <Label forLabel={"total_intl_minutesLbl"} children={"Total Minutes (International)"} />
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_intl_calls"} className={"form-control"} type={"number"} placeholder={"total_intl_calls"} required={"required"} />
                                        <Label forLabel={"total_intl_callsLbl"} children={"Total Calls (International)"} />                                    
                                    </td>
                                    <td className='form-floating'>
                                        <Input id={"total_intl_charge"} className={"form-control"} type={"number"} placeholder={"total_intl_charge"} required={"required"} />
                                        <Label forLabel={"total_intl_chargeLbl"} children={"Total Charge (International)"} />
                                    </td>
                                </tr>
                                <tr>
                                    <td className='form-floating'>
                                        <Input id={"number_customer_service_calls"} className={"form-control"} type={"number"} placeholder={"number_customer_service_calls"} required={"required"} />
                                        <Label forLabel={"number_customer_service_callsLbl"} children={"Number Customer Service Calls"} />                                        
                                    </td>
                                    
                                </tr>
                            </table>

                            <div className='inputButtons text-center'>
                                <Button theme={"danger"} className={"my-3"} children={"Clear Entries"} onClick={handleClear}></Button>
                                <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                            </div>
                            {feature_list.map((feature) => {
                                return <div className='form-floating col-3'>
                                </div>
                            })}
                        </Form>
                        </div>
                </section>

                <section id='result'>
                    <p className='result-label1 text-center'>Prediction Result</p>
                    <p className='result-label2 text-center'>Will the subsciber churn?</p>
                    <p className='result-churn text-center' name='churn-result'>Yes/No</p>
                    <p className='result-label3 text-center'>Prediction Analytics: </p>
                    <div className='data'>
                        <label>Prescision:</label>
                        <label className='result-data' name='precision-result'>0.0000069</label>
                    </div>
                    <div className='data'>
                        <label>Recall:</label>
                        <label className='result-data' name='recall-result'>0.000666</label>
                    </div>
                    <div className='data'>
                        <label>F-Beta:</label>
                        <label className='result-data' name='fbeta-result'>0.000420</label>
                    </div>
                    <div className='resultButton text-center'>
                        <Button theme={"secondary"} className={"my-3"} children={"View Feature Analysis"} onClick={handleClear}></Button>
                        <Button theme={"primary"} className={"my-3"} children={"Input Another Data"} onClick={handleChurn} />
                    </div>
                </section>
            </section>
        </main>

    )
}

export default App