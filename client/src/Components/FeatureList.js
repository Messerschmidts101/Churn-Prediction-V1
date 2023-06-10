import React from 'react'
import InputFeature from './InputFeature'
import Label from './Label'
import { useState, useEffect } from 'react'

function FeatureList({key}) {
    
    const [data, setData] = useState([{}])
    useEffect(() => {
        fetch("/feature_names").then(
            res => res.json()
        ).then(
            data => {
                setData(data)
            }
        )
    }, [])
    let feature_list = []
    data.forEach(feature => feature_list.push(
        <>
            <InputFeature id={"predictionForm"} className={"w-25 form-control d-flex"} key={feature.id} feature={feature.name} />
        </>
    ))
    return (
        <>
            {feature_list}
        </>
    )
}

export default FeatureList