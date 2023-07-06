import React from 'react'
import Header1 from '../components/Header1'

function NotFoundPage() {
    return (
        <>
            <div className='container' style={{ height: "480px" }}>
                <div className='d-flex flex-column text-center h-100'>
                    <div>
                        <Header1 className='display-1 '>404 PAGE DNE</Header1>
                        <p className='display-6'><strong>uwu</strong> Please contact the devs below for this error <strong>uwu</strong></p>
                    </div>
                </div>
            </div>
        </>
        
    )
}

export default NotFoundPage