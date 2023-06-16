import React from 'react'
import Button from '../Components/Button'
import Header1 from '../Components/Header1'
import Header2 from '../Components/Header2'
import Input from '../Components/Input'
import Form from '../Components/Form'
import Label from "../Components/Label"
import InputFeature from '../Components/InputFeature'
import { useState, useEffect } from 'react'

function Home() {

    const [feature_list, setData] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
            }
        )
    }, [])

    const handleScatter = () => {
        var rowValues = document.getElementById("pasteData").value.split("\t"); // Get the pasted row values and split them by tab

        var inputFields = document.querySelectorAll("#predictionForm")[0];
        for (var i = 0; i < inputFields.length; i++) {
            inputFields[i].value = rowValues[i];
        }
    }
    const handleChurn = () => {
        fetch("http://localhost:8080/predict", {
            method: "POST",
            mode: "cors",
            body: JSON.stringify()
        })
    }
    const handleClear = () => {
        feature_list.forEach((feature) => {
            document.querySelector("#"+ feature.name +"\\ predictionValue").value = ""
        })
    }

    return (
        <></>
    )
}

export default Home