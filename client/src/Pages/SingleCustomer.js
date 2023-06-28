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
                data.pop()
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
    const handleChurnColours = () => {
        return (!isNaN(churn) && churn >= 0.50) ? "old-gold" : "brilliant-rose"
    }

    return (
        <>
            <div className='container px-5'>
                <div className='row d-flex flex-row align-items-center pt-5'>
                    <div className='row text-center pt-5'>
                        <Header1 className={""}>Predict a customer's churn!</Header1>
                    </div>
                    <div className='row'>
                        <div className='col form-floating'>
                            <Input id={"pasteData"} className={"form-control"} type="text" placeholder="Paste Excel row here" />
                            <Label forLabel={"pasteData"} children={"Paste Excel row here"} />
                        </div>
                        <Button theme={"primary"} className={"col-md-2 col-sm-3 my-md-2"} children={"Scatter Values"} onClick={handleScatter} />
                    </div>
                </div>
                <div id={"predictionForm"} className={"form row mt-3 flex-sm-row flex-column"}>
                    <Header2 className={""}>...Or input below each value individually</Header2>
                    {feature_list.map((feature) => {
                        return <div className='form-floating col-md-3 col-sm-10 my-md-2'>
                            <InputFeature id={feature.name} className={"form-control d-flex"} key={feature.name} feature={feature.name} />
                            <Label forLabel={feature.name} children={feature.name} />
                        </div>
                    })}
                    <Button theme={"primary"} className={"my-3"} children={"Predict Churn"} onClick={handleChurn} />
                </div>
                <div className='row'>
                    <Button theme={"outline-danger"} className={"my-3"} children={"Clear"} onClick={handleClear} />
                </div>
                <div className='row text-center py-5 unselectable'>
                    <div className='col-md-5 col-sm-7 py-3 bg-dark-green rounded-pill'>
                        <Header2 className={"text-white"}> Churn Rate</Header2>
                        <Header1 className={"display-1 " + (handleChurnColours()) +  " "}>
                        {
                            isNaN(churn) ? "0.00%": (churn * 100).toFixed(2) + "%"
                        }
                        </Header1>
                        <Header1 className={"display-5 " + (handleChurnColours())}>
                        {   
                            (!isNaN(churn) && churn >= 0.50) ? "Churned!" : "Not Churned!"
                        }
                        </Header1>
                    </div>
                </div>
            </div>
        </>
    )
}

export default SingleCustomer