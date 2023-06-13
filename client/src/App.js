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
            <div className='nav'>
                <p className='nav-heading' children={"Churn Prediction"}/>
                <p className='nav-subheading'children={"using Random Forest Classifier"}/>
            </div>
            <div className='container'>
                <Header1 className={"text-center label1"}  children={"Input Your Values"} />
                <div className='d-flex flex-row align-items-center'>
                    <div className='form-floating w-1100'>
                        <Input id={"pasteData"}  className={"form-control d-flex input-row"} type="text" placeholder="Paste Excel row here" />
                        <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    </div>
                    <Button className={"d-flex inline scatter-btn"} children={"Scatter Values"} onClick={handleScatter} />
                </div>
                <Header2 className={"mt-3"} children={"...Or input below each value individually"}></Header2>
                <Form id={"predictionForm"} className={"form row"} action={"prediction"} method={"post"}>
                    {/* <Input id={"state"} className={"form-control"} type={"text"} placeholder={"state"} required={"required"} />
                    <Label forLabel={"state"} children={"State"} /> */}
                    <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                    {feature_list.map((feature) => {
                        return <div className='form-floating col-3'>
                            <InputFeature id={feature.name } className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                            <Label forLabel={feature.name } children={feature.name} />
                        </div>
                    })}
                </Form>
                <Button theme={"outline-danger"} className={"my-3"} children={"Clear"} onClick={handleClear}></Button>
            </div>
            <div className='container'>
                <HighchartsReact highcharts={Highcharts} options={options} />
            </div>
        </main>
    )
}

export default App
