import Button from '../Components/Button'
import Header1 from '../Components/Header1'
import Header2 from '../Components/Header2'
import Input from '../Components/Input'
import Label from "../Components/Label"
import InputFeature from '../Components/InputFeature'
import React from 'react'
import { useState, useEffect } from 'react'

function SingleCustomer() {

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
    let [churn, setChurn] = useState([{}])

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("[type=number]");
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name).value = ""
        })
    }
    const handleChurn = () => {
        let feature_post = []
        let inputFields = document.querySelectorAll("[type=number]");
        for(let i = 0; i < inputFields.length; i++) {
            let name = inputFields[i].id.split(" ")[0]
            let value = inputFields[i].value
            feature_post.push({
                "name": name,
                "value": value
            })
        }
        console.log("feature_post: ", feature_post)

        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ feature_post })
        }).then(
            res => res.json()
        ).then(
            data => {
                console.log(data)
                setChurn(data)
            }
        ).catch(
            e => console.error("No prediction yet\n", e)
        )
        
    }
    

    return (
        <>
            <div className='container'>
                <Header1 className={"row text-center pt-5"}>Predict a customer's churn!</Header1>
                <div className='row d-flex flex-row align-items-center pt-5'>
                    <div className='col form-floating'>
                        <Input id={"pasteData"} className={"form-control"} type="text" placeholder="Paste Excel row here" />
                        <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                    </div>
                    <Button theme={"primary"} className={"col-2"} children={"Scatter Values"} onClick={handleScatter} />
                </div>
                <Header2 className={"row mt-3"} children={"...Or input below each value individually"} />
                <div id={"predictionForm"} className={"form row"}>
                    {feature_list.map((feature) => {
                        return <div className='form-floating col-3'>
                            <InputFeature id={feature.name} className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                            <Label forLabel={feature.name} children={feature.name} />
                        </div>
                    })}
                    <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                </div>
                <div className='row'>
                    <Button theme={"outline-danger"} className={"my-3"} children={"Clear"} onClick={handleClear} />
                </div>
                <div className='row text-center'>
                    <Header2> Churn Rate</Header2>
                    <Header1 className={"display-1"}>
                    {
                        isNaN(churn) ? "0.00%": churn.toFixed(2) + "%"
                    }
                    </Header1>
                </div>
            </div>
        </>
    )
}

export default SingleCustomer